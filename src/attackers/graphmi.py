"""GraphMI variant backed by the shared retriever surrogate."""

"""
GraphMI: Model Inversion Attacks Against Graph Neural Networks
(Adapted for retrieval demotion)

Based on: "Model Inversion Attacks Against Graph Neural Networks"
(Zhang et al., TKDE 2023)

Adaptation: Instead of inverting a classifier (cross-entropy on labels),
we invert the retrieval encoder.  The surrogate GCN is trained to reproduce
the victim retriever's embeddings (MSE loss), and PGD optimises the
adjacency to minimise the embedding-reconstruction loss through that
surrogate.  Edges with high reconstruction probability are the ones most
important for the retrieval pipeline and are prioritised for removal.

Modules (following the paper):
1. Projected Gradient Descent — optimise continuous adjacency via convex
   relaxation, but with a retrieval embedding loss instead of CE.
2. Graph Auto-Encoder — post-process using transferred surrogate weights.
3. Edge scoring — reconstruction probabilities → removal priority.
""" 

from src.attackers.base import Attacker
from torch_geometric.data import Data
from typing import Tuple, Dict
from omegaconf import DictConfig
from src.retrievers.base import Retriever
from src.attackers.surrogate_retriever import get_or_train_surrogate_retriever
from src.utils.logger_utils import setup_logger
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = setup_logger(__name__)



class SurrogateEncoder(nn.Module):
    """2-layer GCN that outputs embeddings matching the retriever dim."""

    def __init__(self, nfeat: int, nhid: int, nemb: int, dropout: float = 0.5):
        super().__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nemb)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        h = F.relu(adj_norm @ x @ self.W1.weight.t() + self.W1.bias)
        h = F.dropout(h, self.dropout, training=self.training)
        out = adj_norm @ h @ self.W2.weight.t() + self.W2.bias
        return out

class EmbeddingGCN(nn.Module):
    """Single-layer GCN encoder for graph auto-encoder module."""

    def __init__(self, nfeat: int, nhid: int):
        super().__init__()
        self.W = nn.Linear(nfeat, nhid)

    def forward(self, x, adj_norm):
        return F.relu(adj_norm @ x @ self.W.weight.t() + self.W.bias)

class GraphMIAttack(Attacker):
    """
    GraphMI adapted for retrieval demotion.

    1. Obtain target embeddings Z* from the victim retriever.
    2. Train a differentiable surrogate GCN to reproduce Z* (MSE).
    3. PGD: optimise A_recon to minimise
           L_emb(A) + α·L_smooth(A) + β·‖a‖₂
       where L_emb = ‖surrogate(X, A) − Z*‖² (cosine or MSE).
    4. Graph auto-encoder post-processing with transferred weights.
    5. Score edges by reconstruction probability.
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

        # PGD hyper-parameters
        self.inversion_epochs = getattr(cfg, "inversion_epochs", 100)
        self.lr = getattr(cfg, "learning_rate", 0.1)
        self.alpha = getattr(cfg, "alpha", 0.001)       # feature-smoothness weight
        self.beta = getattr(cfg, "beta", 0.0001)        # sparsity weight
        self.nhid = getattr(cfg, "nhid", 128)

        # Surrogate training hyper-parameters
        self.surrogate_epochs = getattr(cfg, "surrogate_epochs", 200)
        self.surrogate_lr = getattr(cfg, "surrogate_lr", 0.01)
        self.surrogate_wd = getattr(cfg, "surrogate_weight_decay", 5e-4)

        self.use_autoencoder = getattr(cfg, "use_autoencoder", True)
        self.emb_loss_type = getattr(cfg, "emb_loss_type", "cosine")  # "mse" or "cosine"

        # Higher score → higher removal priority
        self.sort_reverse = True

        self.retriever = retriever

        # Run one GraphMI pipeline per trial so each trial can use a separately
        # initialized retriever surrogate.
        self.retriever_surrogate_time = 0.0
        self.edge_probs_runs = []
        for run_idx in range(self.trials):
            self._surrogate_run_id = f"run{run_idx + 1}" if self.trials > 1 else None
            self._surrogate_seed = self.seed + run_idx
            self.edge_probs_runs.append(self._run_graphmi())
        self.edge_probs = self.edge_probs_runs[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
        """D^{-1/2} A D^{-1/2}"""
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    def _vec_to_symmetric(self, vec: torch.Tensor, n: int) -> torch.Tensor:
        """Flat upper-triangle vector → symmetric N×N matrix."""
        mat = torch.zeros(n, n, device=vec.device)
        idx = torch.triu_indices(n, n, offset=1)
        mat[idx[0], idx[1]] = vec
        mat = mat + mat.t()
        return mat

    # ------------------------------------------------------------------
    # Step 1 — target embeddings from the victim retriever
    # ------------------------------------------------------------------

    def _get_target_embeddings(self) -> torch.Tensor:
        """Return Z* = retriever.encode(X, A)  (detached)."""
        with torch.no_grad():
            z = self.retriever.encode(self.data.x, self.data.edge_index)
        return z.detach()

    # ------------------------------------------------------------------
    # Step 2 — train a differentiable surrogate to mimic the retriever
    # ------------------------------------------------------------------

    def _train_surrogate(self, z_target: torch.Tensor):
        """
        Train a 2-layer GCN to minimise ‖surrogate(X,A) − Z*‖²
        on the original graph so that we have a differentiable proxy
        of the retriever.
        """
        data = self.data
        n = data.num_nodes
        nfeat = data.x.size(1)
        nemb = z_target.size(1)

        # Dense adjacency + self-loops + sym-norm (original graph)
        adj = torch.zeros(n, n, device=self.device)
        ei = data.edge_index
        adj[ei[0], ei[1]] = 1.0
        adj_hat = adj + torch.eye(n, device=self.device)
        adj_norm = self._symmetric_normalize(adj_hat)

        model = SurrogateEncoder(nfeat, self.nhid, nemb, dropout=0.0).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.surrogate_lr, weight_decay=self.surrogate_wd
        )

        best_loss = float("inf")
        best_state = None
        patience, patience_ctr = 30, 0

        for epoch in range(self.surrogate_epochs):
            model.train()
            optimizer.zero_grad()
            z_pred = model(data.x, adj_norm)

            # MSE between predicted and target embeddings
            loss = F.mse_loss(z_pred, z_target)
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

        # Report cosine similarity between surrogate and target
        model.eval()
        with torch.no_grad():
            z_pred = model(data.x, adj_norm)
            cos = F.cosine_similarity(z_pred, z_target, dim=1).mean().item()
        logger.info(
            f"Surrogate trained: MSE={best_loss:.6f}, "
            f"avg_cos_sim={cos:.4f} (epoch {epoch+1})"
        )
        return model, adj_norm

    # ------------------------------------------------------------------
    # Step 3 — PGD inversion (retrieval-based loss)
    # ------------------------------------------------------------------

    def _embedding_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Retrieval-aware reconstruction loss.
        - "cosine": 1 − mean(cos_sim)  — directly matches the retrieval
          ranking metric (cosine similarity).
        - "mse":  mean ‖z_pred − z_target‖²
        """
        if self.emb_loss_type == "cosine":
            return 1.0 - F.cosine_similarity(z_pred, z_target, dim=1).mean()
        return F.mse_loss(z_pred, z_target)

    def _compute_feature_smoothness(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """tr(X^T L_hat X) with normalised Laplacian."""
        deg = adj.sum(dim=1).clamp(min=1e-8)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        x_norm = x * deg_inv_sqrt.unsqueeze(1)
        a_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return torch.trace(x_norm.t() @ x_norm) - torch.trace(x_norm.t() @ a_norm @ x_norm)

    def _projection(self, adj_changes: torch.Tensor, num_edges: int):
        """Project onto simplex {a ∈ [0,1]^n : sum(a) ≤ num_edges}."""
        values = adj_changes.data.clamp(0, 1)
        if values.sum() <= num_edges:
            return
        lo, hi = 0.0, values.max().item()
        for _ in range(50):
            mid = (lo + hi) / 2
            if (values - mid).clamp(0, 1).sum() > num_edges:
                lo = mid
            else:
                hi = mid
        adj_changes.data = (adj_changes.data - lo).clamp(0, 1)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def _run_graphmi(self) -> torch.Tensor:
        """
        1. Obtain target embeddings from the retriever.
        2. Train differentiable surrogate.
        3. PGD with retrieval embedding loss.
        4. Graph auto-encoder post-processing.
        Returns N×N edge probability matrix.
        """
        data = self.data
        n = data.num_nodes
        x = data.x

        if hasattr(self, "_surrogate_seed") and self._surrogate_seed is not None:
            random.seed(int(self._surrogate_seed))
            np.random.seed(int(self._surrogate_seed))
            torch.manual_seed(int(self._surrogate_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self._surrogate_seed))

        logger.info(f"Running GraphMI (retrieval mode) on {n} nodes …")

        # --- target embeddings ---
        z_target = self._get_target_embeddings()
        logger.info(f"Target embedding dim = {z_target.size(1)}")

        # --- surrogate ---
        surrogate, _ = self._train_surrogate(z_target)
        for p in surrogate.parameters():
            p.requires_grad = False

        # --- PGD ---
        n_vars = n * (n - 1) // 2
        adj_changes = torch.zeros(n_vars, device=self.device, requires_grad=True)

        num_edges_orig = data.edge_index.size(1) // 2
        density = num_edges_orig / (n * (n - 1) / 2)
        num_edges_target = int(density * n_vars)

        logger.info(
            f"PGD: {self.inversion_epochs} epochs, density={density:.6f}, "
            f"target_edges={num_edges_target}, loss={self.emb_loss_type}"
        )

        for epoch in range(self.inversion_epochs):
            adj_recon = self._vec_to_symmetric(adj_changes, n)
            adj_hat = adj_recon + torch.eye(n, device=self.device)
            adj_norm = self._symmetric_normalize(adj_hat)

            # Forward through frozen surrogate
            z_pred = surrogate(x, adj_norm)

            # --- Retrieval embedding loss ---
            loss_emb = self._embedding_loss(z_pred, z_target)

            # Sparsity
            loss_sparse = self.beta * torch.norm(adj_changes, p=2)

            # Feature smoothness (enabled after epoch 50, same as paper)
            loss_smooth = torch.tensor(0.0, device=self.device)
            if epoch >= 50:
                loss_smooth = self.alpha * self._compute_feature_smoothness(adj_recon, x)

            loss = loss_emb + loss_sparse + loss_smooth

            grad = torch.autograd.grad(loss, adj_changes)[0]

            with torch.no_grad():
                adj_changes.data -= self.lr * grad
                adj_changes.data.clamp_(0, 1)
                self._projection(adj_changes, num_edges_target)

            if (epoch + 1) % 25 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.inversion_epochs}: "
                    f"L_emb={loss_emb.item():.4f}  "
                    f"L_sparse={loss_sparse.item():.4f}  "
                    f"L_smooth={loss_smooth.item():.4f}"
                )

        # --- Graph auto-encoder post-processing ---
        with torch.no_grad():
            if self.use_autoencoder:
                adj_recon = self._vec_to_symmetric(adj_changes, n)
                adj_hat = adj_recon + torch.eye(n, device=self.device)
                adj_norm = self._symmetric_normalize(adj_hat)

                encoder = EmbeddingGCN(x.size(1), self.nhid).to(self.device)
                encoder.W.weight.data = surrogate.W1.weight.data.clone()
                encoder.W.bias.data = surrogate.W1.bias.data.clone()

                z = encoder(x, adj_norm)
                z_norm = F.normalize(z, p=2, dim=1)
                edge_probs = torch.sigmoid(z_norm @ z_norm.t())
            else:
                edge_probs = self._vec_to_symmetric(adj_changes.data, n)

        edge_probs.fill_diagonal_(0)
        edge_probs = (edge_probs + edge_probs.t()) / 2

        logger.info(
            f"GraphMI complete. Edge-prob range: "
            f"[{edge_probs.min().item():.4f}, {edge_probs.max().item():.4f}]"
        )
        return edge_probs

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def score_edges(self, target_node: int) -> Dict[Tuple[int, int], float]:
        """
        Higher reconstruction probability → more important for retrieval
        → higher removal priority.
        """
        target_edges = self.get_target_edges(target_node)
        return {(i, j): self.edge_probs[i, j].item() for i, j in target_edges}

    def attack(self, target_node: int, budget: int, add_edge: bool = False) -> list[Data]:
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        if add_edge:
            saved_trials = self.trials
            self.trials = 1
            try:
                return self._attack_add(target_node, budget)
            finally:
                self.trials = saved_trials
        embedding_list = []
        for run_idx in range(self.trials):
            torch.manual_seed(self.seed + run_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed + run_idx)
            random.seed(self.seed + run_idx)
            np.random.seed(self.seed + run_idx)

            self.edge_probs = self.edge_probs_runs[run_idx]
            scores = self.score_edges(target_node)
            sorted_edges = sorted(scores.keys(), key=lambda e: scores[e], reverse=self.sort_reverse)
            edges_to_remove = sorted_edges[:budget]
            embedding_list.append(self._remove_edges(edges_to_remove))
        self.edge_probs = self.edge_probs_runs[0]
        return embedding_list
    

class GraphMIAttackRetriever(GraphMIAttack): 
    """GraphMI using the common SurrogateRetrieverGCN helper.

    This keeps GraphMI's original retrieval-adapted logic, but avoids a second
    local surrogate implementation. It is intentionally very close to
    graphmi.py, because graphmi.py was already using the correct idea: train a
    GCN to reproduce retriever embeddings.
    """

    def _train_surrogate(self, z_target):
        cache_dir = getattr(self.system_config.general, "cache_dir", "cache")
        data_name = self.system_config.dataset.name
        retriever_name = self.system_config.retriever.name
        graph_model = getattr(self.system_config.general, "graph_model", "gcn")

        model, _ = get_or_train_surrogate_retriever(
            self.data,
            self.retriever,
            data_name,
            retriever_name,
            graph_model=graph_model,
            cache_dir=cache_dir,
            hidden_channels=self.nhid,
            epochs=self.surrogate_epochs,
            lr=self.surrogate_lr,
            weight_decay=self.surrogate_wd,
            dropout=0.0,
            loss_type=self.emb_loss_type,
            patience=getattr(self.cfg, "surrogate_patience", 30),
            device=str(self.device),
            run_id=getattr(self, "_surrogate_run_id", None),
            seed=getattr(self, "_surrogate_seed", None),
        )
        self.retriever_surrogate_time += float(getattr(model, "retriever_surrogate_time", 0.0))
        logger.info(
            f"GraphMI-Retriever loaded retriever surrogate run={getattr(self, '_surrogate_run_id', None)} "
            f"time={getattr(model, 'retriever_surrogate_time', 0.0):.4f}s"
        )
        return model, None


    def score_candidates(self, target_node):
        """Add-edge mode: score candidate new neighbors by GraphMI's reconstruction
        plausibility matrix (``edge_probs``), in the DEMOTION (reverse) direction.

        GraphMI's PGD reconstructs an edge-probability matrix; a high value means
        the endpoints are reconstruction-plausible (similar in the inverted
        surrogate space). Demotion needs the target connected to DISSIMILAR nodes,
        so the default ``mi_direction="reverse"`` returns the NEGATIVE plausibility
        (add least-plausible non-edges first); ``"forward"`` adds the most-plausible
        (a stealth/plausibility floor). Deterministic given the inversion, so
        add-edge runs a single trial.
        """
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        direction = str(getattr(self, "mi_direction", getattr(self.cfg, "mi_direction", "reverse"))).lower()
        ep = self.edge_probs_runs[0]
        row = ep[target_node].float().to(self.device)
        return -row if direction == "reverse" else row

