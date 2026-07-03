"""
GraphZOOM: Subgraph Black-Box Attack Against Inductive GNNs
(Adapted for retrieval demotion)

Based on: "GraphZOOM: subgraph black-box attack against inductive graph
neural networks" (Tang et al., Complex & Intelligent Systems 2026)

Adaptation: Instead of perturbing neighbor features to flip a classification
label, we use GraphZOOM's zeroth-order probing mechanism to score edges
for removal priority.  For each edge (target, neighbor), we measure how
sensitive the retrieval embeddings are to that neighbor's features by
setting them to extremes (all-1s / all-0s) and measuring the embedding
difference.  Optionally, a ZOO finite-difference refinement estimates
per-dimension gradient magnitudes for the top-K most influential edges.

Two modes:
  - use_surrogate=false (default): probes the retriever passed in by the harness.
    Under the black-box-transfer wiring (load_attackers) that is the SURROGATE
    retriever — NOT the victim — so this is a transfer attack consistent with the
    other gradient baselines (the victim is only the eval target).
  - use_surrogate=true: trains a SEPARATE label-supervised GCN (on the attack-split
    TRAIN ids only — never the test targets) and probes it instead. Non-default;
    uses node labels, unlike the label-free default.

Edges with higher influence scores are prioritised for removal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from torch_geometric.data import Data
from omegaconf import DictConfig

from src.attackers.base import Attacker
from src.retrievers.base import Retriever
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# ----------------------------------------------------------------------
# Surrogate model (used when use_surrogate=true)
# Trained from scratch on node classification — no victim queries.
# ----------------------------------------------------------------------

class _SurrogateGCN(nn.Module):
    """2-layer GCN trained on node classification.

    forward()  returns class logits (for training).
    embed()    returns the hidden-layer representations (for ZOO probing).
    """

    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float = 0.5):
        super().__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    @staticmethod
    def _sym_norm(adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(adj_norm @ x @ self.W1.weight.t() + self.W1.bias)
        h = F.dropout(h, self.dropout, training=self.training)
        out = adj_norm @ h @ self.W2.weight.t() + self.W2.bias
        return out

    def embed(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Return hidden-layer representations (no classification head)."""
        h = F.relu(adj_norm @ x @ self.W1.weight.t() + self.W1.bias)
        return h


class GraphZOOMAttack(Attacker):
    """
    GraphZOOM adapted for retrieval demotion via edge scoring.

    Phase 1 — Coarse probing (always runs):
        For each edge (target, u), set u's features to all-1s and all-0s,
        encode via the encoder, and measure the embedding difference.

    Phase 2 — ZOO refinement (optional):
        For the top-K edges from Phase 1, estimate the finite-difference
        gradient of the retrieval objective w.r.t. the neighbor's features
        on a random subset of dimensions.  The L2 norm of this gradient
        vector becomes the refined score.

    The encoder is either the victim retriever (use_surrogate=false)
    or a trained surrogate GCN (use_surrogate=true).
    """

    def __init__(
        self,
        cfg: DictConfig,
        system_config: DictConfig,
        data: Data,
        retriever: Retriever = None,
        **kwargs,
    ):
        super().__init__(cfg, system_config, data)

        # Higher score → higher removal priority
        self.sort_reverse = True

        self.retriever = retriever

        # Phase 1 config
        self.objective_type = getattr(cfg, "objective_type", "neighbor_embedding")

        # Phase 2 config
        self.use_zoo_refinement = getattr(cfg, "use_zoo_refinement", False)
        self.top_k_refine = getattr(cfg, "top_k_refine", 10)
        self.step_size_h = getattr(cfg, "step_size_h", 1e-4)
        self.n_coords = getattr(cfg, "n_coords", 20)

        # ZOO ADAM parameters (used when zoo refinement is iterative)
        self.zoo_lr = getattr(cfg, "zoo_lr", 0.01)
        self.zoo_itermax = getattr(cfg, "zoo_itermax", 50)
        self.zoo_beta1 = getattr(cfg, "zoo_beta1", 0.9)
        self.zoo_beta2 = getattr(cfg, "zoo_beta2", 0.999)

        # Surrogate config
        self.use_surrogate = getattr(cfg, "use_surrogate", False)
        self.surrogate_nhid = getattr(cfg, "surrogate_nhid", 128)
        self.surrogate_epochs = getattr(cfg, "surrogate_epochs", 200)
        self.surrogate_lr = getattr(cfg, "surrogate_lr", 0.01)
        self.surrogate_wd = getattr(cfg, "surrogate_weight_decay", 5e-4)
        self._score_cache: Dict[int, Dict[Tuple[int, int], float]] = {}

        # Build surrogate or keep retriever as encoder
        self.surrogate = None
        self.adj_norm = None
        if self.use_surrogate:
            # Train GCN from scratch on node classification — no victim queries
            self._build_surrogate()
            self.original_embedding = self._surrogate_encode(data.x.to(self.device))
        else:
            # Direct mode: use the victim retriever throughout
            with torch.no_grad():
                self.original_embedding = self.retriever.encode(
                    data.x.to(self.device), data.edge_index.to(self.device)
                ).detach()

        mode = "surrogate" if self.use_surrogate else "direct"
        logger.info(
            f"GraphZOOM initialised: mode={mode}, objective={self.objective_type}, "
            f"zoo_refine={self.use_zoo_refinement}, "
            f"top_k={self.top_k_refine}, n_coords={self.n_coords}"
        )

    # ------------------------------------------------------------------
    # Surrogate training
    # ------------------------------------------------------------------

    def _build_surrogate(self):
        """Train a 2-layer GCN from scratch on node classification.

        Uses graph structure, node features, and labels only.
        No victim retriever queries — fully independent surrogate.
        Hidden-layer representations are used as embeddings for ZOO probing.
        """
        data = self.data
        x = data.x.to(self.device)
        y = data.y.to(self.device)
        n = data.num_nodes
        nfeat = x.size(1)
        nclass = int(y.max().item()) + 1

        # Dense adjacency with self-loops + symmetric normalisation
        adj = torch.zeros(n, n, device=self.device)
        ei = data.edge_index.to(self.device)
        adj[ei[0], ei[1]] = 1.0
        adj_hat = adj + torch.eye(n, device=self.device)
        adj_norm = _SurrogateGCN._sym_norm(adj_hat)

        # Train the surrogate ONLY on the attack-split TRAIN ids (disjoint from the test targets),
        # so no attack target's label leaks in. The old fallback (torch.ones) trained on EVERY node
        # including the test targets on datasets without a train_mask (Amazon/Coauthor/CoraFull/DBLP)
        # — label + target leakage (audit M4). Mirrors InfAttack's leakage-safe LR fit.
        from src.data.splits import load_or_create_attack_split
        train_ids, _ = load_or_create_attack_split(
            data, self.system_config.dataset.name,
            getattr(self.system_config.general, "cache_dir", "cache"),
            seed=getattr(self.system_config.general, "seed", 42))
        train_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        train_mask[torch.tensor(train_ids, dtype=torch.long, device=self.device)] = True

        model = _SurrogateGCN(nfeat, self.surrogate_nhid, nclass, dropout=0.5).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.surrogate_lr, weight_decay=self.surrogate_wd
        )

        best_loss = float("inf")
        best_state = None
        patience, patience_ctr = 30, 0

        for epoch in range(self.surrogate_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x, adj_norm)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits = model(x, adj_norm)
            preds = logits[train_mask].argmax(dim=1)
            acc = (preds == y[train_mask]).float().mean().item()
        logger.info(
            f"Surrogate trained from scratch: CE_loss={best_loss:.4f}, "
            f"train_acc={acc:.4f} (epoch {epoch + 1})"
        )

        self.surrogate = model
        self.adj_norm = adj_norm

    def _surrogate_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode via the surrogate GCN's hidden layer (not classification logits)."""
        with torch.no_grad():
            return self.surrogate.embed(x.to(self.device), self.adj_norm)

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Route to surrogate or victim retriever."""
        if self.use_surrogate:
            return self._surrogate_encode(x)
        with torch.no_grad():
            return self.retriever.encode(x.to(self.device), edge_index.to(self.device))

    # ------------------------------------------------------------------
    # Phase 1 — coarse max/min probing  (Eq. 3 adaptation)
    # ------------------------------------------------------------------

    def _probe_neighbor_influence(
        self, target_node: int, neighbor: int
    ) -> float:
        """
        Measure how much the embeddings change when *neighbor*'s
        features are set to all-ones vs all-zeros (GraphZOOM Eq. 3).

        Returns a scalar influence score (higher = more influential).
        """
        x = self.data.x.to(self.device).clone()
        edge_index = self.data.edge_index.to(self.device)
        d = x.size(1)

        # Save original features of the neighbor
        orig_features = x[neighbor].clone()

        # --- Gmax: neighbor features = 1 ---
        x[neighbor] = torch.ones(d, device=self.device, dtype=x.dtype)
        emb_max = self._encode(x, edge_index)

        # --- Gmin: neighbor features = 0 ---
        x[neighbor] = torch.zeros(d, device=self.device, dtype=x.dtype)
        emb_min = self._encode(x, edge_index)

        # Restore original features
        x[neighbor] = orig_features

        # Compute score based on objective type
        if self.objective_type == "neighbor_embedding":
            score = (emb_max[neighbor] - emb_min[neighbor]).norm().item()
        elif self.objective_type == "target_embedding":
            score = (emb_max[target_node] - emb_min[target_node]).norm().item()
        elif self.objective_type == "all_embedding":
            score = (emb_max - emb_min).norm(dim=1).mean().item()
        else:
            raise ValueError(f"Unknown objective_type: {self.objective_type}")

        return score

    # ------------------------------------------------------------------
    # Phase 2 — ZOO finite-difference refinement
    # ------------------------------------------------------------------

    def _zoo_refine_edge(
        self, target_node: int, neighbor: int
    ) -> float:
        """
        Estimate the L2 norm of the gradient of the retrieval objective
        w.r.t. the neighbor's features using finite differences on a
        random subset of dimensions.

        Objective: L2 distance between the perturbed and original
        embedding of *neighbor*.
        """
        x = self.data.x.to(self.device).clone()
        edge_index = self.data.edge_index.to(self.device)
        d = x.size(1)
        h = self.step_size_h

        orig_features = x[neighbor].clone()
        orig_emb_neighbor = self.original_embedding[neighbor]

        # Sample dimensions to probe
        n_dims = min(self.n_coords, d) if self.n_coords > 0 else d
        dims = np.random.choice(d, size=n_dims, replace=False)

        grad_sq_sum = 0.0

        for j in dims:
            # +h perturbation
            x[neighbor] = orig_features.clone()
            x[neighbor, j] += h
            emb_plus = self._encode(x, edge_index)
            obj_plus = (emb_plus[neighbor] - orig_emb_neighbor).norm().item()

            # -h perturbation
            x[neighbor] = orig_features.clone()
            x[neighbor, j] -= h
            emb_minus = self._encode(x, edge_index)
            obj_minus = (emb_minus[neighbor] - orig_emb_neighbor).norm().item()

            grad_j = (obj_plus - obj_minus) / (2 * h)
            grad_sq_sum += grad_j ** 2

        # Restore
        x[neighbor] = orig_features

        # Scale up if we only sampled a subset of dimensions
        if n_dims < d:
            grad_sq_sum *= d / n_dims

        return float(np.sqrt(grad_sq_sum))

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def score_edges(self, target_node: int) -> Dict[Tuple[int, int], float]:
        """
        Score each edge connected to *target_node*.

        Higher score → higher removal priority (sort_reverse = True).
        """
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        target_node = int(target_node)
        cached_scores = self._score_cache.get(target_node)
        if cached_scores is not None:
            return cached_scores

        target_edges = self.get_target_edges(target_node)

        if len(target_edges) == 0:
            return {}

        # --- Phase 1: coarse probing ---
        coarse_scores: Dict[Tuple[int, int], float] = {}
        for edge in target_edges:
            neighbor = self.get_edge_neighbor(edge, target_node)
            coarse_scores[edge] = self._probe_neighbor_influence(
                target_node, neighbor
            )

        # --- Phase 2: optional ZOO refinement on top-K ---
        if not self.use_zoo_refinement:
            self._score_cache[target_node] = coarse_scores
            return coarse_scores

        sorted_edges = sorted(
            coarse_scores.keys(),
            key=lambda e: coarse_scores[e],
            reverse=True,
        )
        top_k = sorted_edges[: self.top_k_refine]

        final_scores: Dict[Tuple[int, int], float] = {}
        for edge in target_edges:
            if edge in top_k:
                neighbor = self.get_edge_neighbor(edge, target_node)
                final_scores[edge] = self._zoo_refine_edge(
                    target_node, neighbor
                )
            else:
                final_scores[edge] = coarse_scores[edge]

        self._score_cache[target_node] = final_scores
        return final_scores
