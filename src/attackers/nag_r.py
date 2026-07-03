"""NAG-R variant using a retriever-embedding surrogate."""

"""
NAG-R: Nesterov Accelerated Gradient attack on GNNs.

Faithful adaptation of:
"A Black-box Adversarial Attack Method via Nesterov Accelerated Gradient
and Rewiring Towards Attacking Graph Neural Networks"
(Zhao et al., IEEE TBD 2023)

Reference:
https://github.com/AHU-ICKE/Network-Representation-Learning/tree/main/NAG-R/Code

The released NGA code (attack/nga.py) is removal-only — the add-edge
branch is commented out. We follow that convention here since this
framework is edge-removal only. Per greedy step:

  1. Compute D^{-1/2} (A+I) D^{-1/2}.
  2. Nesterov look-ahead: A_norm_nes = A_norm + decay*step*momentum,
     where momentum is non-zero only in row `target_node` (matches the
     original code, which only ever updates momentum[target_node, :]).
  3. Surrogate forward, NLL/CE loss for target.
  4. Take dL/dA at row `target_node`, L1-normalize, accumulate momentum
     (NAG update: grad <- decay*momentum + grad).
  5. Pick neighbor with largest |grad| s.t. sign(grad) < 0 — i.e. lowering
     A[target, k] increases loss.
  6. Zero edge_values for both directions; repeat.

Memory: sparse adjacency with edge_values (length E), like AGPOA, to
avoid the O(N^2) dense storage that would OOM on coauthor-physics.
The look-ahead — a length-N delta — is injected only at the target row's
propagation in both GCN layers (algebraically identical to dense).

Faithfulness note: original NGA reads grad[target_node, :] only — the
target's row of dL/dA. We follow that exactly (out-edges only); we do
NOT sum incoming-edge gradients the way AGPOA does.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data
from typing import Dict, List, Tuple

from src.attackers.base import Attacker
from src.oracle.query import neighborhood_centroid_query
from src.attackers.surrogate_gcn import get_or_train_surrogate_gcn
from src.attackers.surrogate_retriever import get_or_train_surrogate_retriever 
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class NAGRAttack(Attacker):

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        super().__init__(cfg, system_config, data)

        # NAG hyper-params; defaults match the original repo.
        # Cora-tuned per the source: decay=0.6, step=0.95.
        self.decay = float(getattr(cfg, "decay", 0.9))
        self.step = float(getattr(cfg, "step", 0.7))

        # High score = picked earlier; base.attack() with sort_reverse=True
        # then removes highest-scored edges first.
        self.sort_reverse = True

        cache_dir = getattr(system_config.general, "cache_dir", "cache")
        data_name = system_config.dataset.name
        self.surrogate = get_or_train_surrogate_gcn(
            data, data_name, cache_dir=cache_dir, device=str(self.device)
        )
        self.surrogate.eval()
        for p in self.surrogate.parameters():
            p.requires_grad_(False)

        self._W1 = self._extract_weight(self.surrogate.conv1)
        self._b1 = self._extract_bias(self.surrogate.conv1)
        self._W2 = self._extract_weight(self.surrogate.conv2)
        self._b2 = self._extract_bias(self.surrogate.conv2)

        n = data.num_nodes
        self._n = n
        ei = data.edge_index.to(self.device)
        no_self = ei[0] != ei[1]
        self._edge_index = ei[:, no_self]
        self._E = self._edge_index.shape[1]

        self._x = data.x.to(self.device)
        self._labels = data.y.to(self.device)
        self._xW1 = (self._x @ self._W1.t()).detach()

        self_idx = torch.arange(n, device=self.device)
        self._self_loop_indices = torch.stack([self_idx, self_idx])

        self._pseudo_labels = self._compute_pseudo_labels(data)

        self._order_cache: Dict[int, List[Tuple[int, int]]] = {}

        logger.info(
            f"NAG-R initialized (sparse): decay={self.decay}, step={self.step}, "
            f"E={self._E}, N={n}"
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_weight(conv):
        if hasattr(conv, "lin"):
            return conv.lin.weight.detach()
        return conv.weight.detach()

    @staticmethod
    def _extract_bias(conv):
        if hasattr(conv, "bias") and conv.bias is not None:
            return conv.bias.detach()
        return None

    def _compute_pseudo_labels(self, data: Data) -> torch.Tensor:
        """Surrogate predictions, overridden by ground-truth on train nodes."""
        with torch.no_grad():
            out = self.surrogate(
                data.x.to(self.device), data.edge_index.to(self.device)
            )
            pseudo = out.argmax(dim=1)
        if hasattr(data, "train_mask") and data.train_mask is not None:
            tm = data.train_mask.to(self.device)
            pseudo[tm] = self._labels[tm]
        return pseudo

    # ------------------------------------------------------------------
    # Differentiable sparse GCN forward with target-row Nesterov look-ahead
    # ------------------------------------------------------------------

    def _gcn_forward_with_lookahead(
        self,
        edge_values: torch.Tensor,
        target_node: int,
        delta_target_row: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse 2-layer GCN forward.

        delta_target_row is decay*step*momentum[target_node, :] in the
        original dense formulation. Adding it to A_norm[target, :] only
        changes propagation contributions at the target row; we inject
        that delta into both layers' target-row outputs directly.
        """
        n = self._n

        all_indices = torch.cat([self._edge_index, self._self_loop_indices], dim=1)
        all_values = torch.cat(
            [edge_values, torch.ones(n, device=self.device)]
        )

        adj_hat = torch.sparse_coo_tensor(all_indices, all_values, (n, n))
        ones_col = torch.ones(n, 1, device=self.device)
        deg = torch.sparse.mm(adj_hat, ones_col).squeeze(1)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = torch.where(
            torch.isinf(deg_inv_sqrt),
            torch.zeros_like(deg_inv_sqrt),
            deg_inv_sqrt,
        )

        row, col = all_indices
        norm_values = deg_inv_sqrt[row] * all_values * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(all_indices, norm_values, (n, n))

        # Layer 1
        h_base = torch.sparse.mm(adj_norm, self._xW1)
        delta_h_target = delta_target_row @ self._xW1
        h = h_base.clone()
        h[target_node] = h[target_node] + delta_h_target
        if self._b1 is not None:
            h = h + self._b1
        h = F.relu(h)

        # Layer 2
        h_W2 = h @ self._W2.t()
        z_base = torch.sparse.mm(adj_norm, h_W2)
        delta_z_target = delta_target_row @ h_W2
        out = z_base.clone()
        out[target_node] = out[target_node] + delta_z_target
        if self._b2 is not None:
            out = out + self._b2

        return out

    def _compute_target_grad(
        self,
        target_node: int,
        edge_values_current: torch.Tensor,
        momentum_target_row: torch.Tensor,
    ) -> torch.Tensor:
        """dL/d(edge_values) for target-node loss with NAG look-ahead."""
        edge_values = (
            edge_values_current.detach().clone().requires_grad_(True)
        )
        delta_target_row = (self.decay * self.step * momentum_target_row).detach()
        out = self._gcn_forward_with_lookahead(
            edge_values, target_node, delta_target_row
        )
        loss = F.cross_entropy(
            out[target_node : target_node + 1],
            self._pseudo_labels[target_node : target_node + 1],
        )
        (grad,) = torch.autograd.grad(loss, edge_values)
        return grad.detach()

    # ------------------------------------------------------------------
    # Greedy NAG attack
    # ------------------------------------------------------------------

    def _compute_attack_order(
        self, target_node: int, max_budget: int
    ) -> List[Tuple[int, int]]:
        """Greedy edge removal guided by NAG-accelerated gradient at target row."""
        edge_values = torch.ones(self._E, device=self.device)

        # Map neighbor -> position in edge_index for outgoing/incoming edges.
        out_mask = self._edge_index[0] == target_node
        in_mask = self._edge_index[1] == target_node
        out_positions = out_mask.nonzero(as_tuple=True)[0]
        in_positions = in_mask.nonzero(as_tuple=True)[0]
        out_neighbors = self._edge_index[1, out_positions]
        in_neighbors = self._edge_index[0, in_positions]

        nb_to_out = {int(j): int(k) for k, j in zip(out_positions, out_neighbors)}
        nb_to_in = {int(j): int(k) for k, j in zip(in_positions, in_neighbors)}
        if self.edge_direction == "outgoing":
            nb_to_in = {}
        elif self.edge_direction == "incoming":
            nb_to_out = {}

        if not nb_to_out and not nb_to_in:
            return []

        # Original keeps momentum as N×N but only writes target's row, so
        # a length-N vector suffices.
        momentum_target_row = torch.zeros(self._n, device=self.device)

        removed: set = set()
        perturbations: List[Tuple[int, int]] = []

        for _ in range(max_budget):
            grad_edges = self._compute_target_grad(
                target_node, edge_values, momentum_target_row
            )

            # Project edge-level gradients to the target's row (out-edges only,
            # matching original's grad[target_node, :]).
            grad_row = torch.zeros(self._n, device=self.device)
            if out_positions.numel() > 0:
                grad_row.index_add_(
                    0, out_neighbors, grad_edges[out_positions]
                )

            # NAG update: L1-normalize then accumulate.
            grad_l1 = grad_row.abs().sum()
            if grad_l1 > 0:
                grad_row = grad_row / grad_l1
            grad_row = momentum_target_row * self.decay + grad_row
            momentum_target_row = grad_row.detach()

            # Pick edge with largest |grad| AND sign(grad) < 0 AND edge exists.
            grad_sort = torch.argsort(grad_row.abs(), descending=True)
            best_j = -1
            best_edge = None
            for k in grad_sort:
                k_int = int(k)
                if k_int == target_node or k_int in removed:
                    continue
                if k_int not in nb_to_out and k_int not in nb_to_in:
                    continue
                if grad_row[k_int].item() < 0:
                    best_j = k_int
                    if k_int in nb_to_in:
                        best_edge = (k_int, target_node)
                    else:
                        best_edge = (target_node, k_int)
                    break

            if best_j == -1:
                break

            if best_j in nb_to_out:
                edge_values[nb_to_out[best_j]] = 0
            if best_j in nb_to_in:
                edge_values[nb_to_in[best_j]] = 0

            removed.add(best_j)
            perturbations.append(best_edge)

        return perturbations

    # ------------------------------------------------------------------
    # Attacker interface
    # ------------------------------------------------------------------

    def score_edges(
        self, target_node: int
    ) -> Dict[Tuple[int, int], float]:
        if target_node not in self._order_cache:
            max_b = getattr(self.system_config.general, "budget", 5)
            self._order_cache[target_node] = self._compute_attack_order(
                target_node, max_b
            )
        ordered = self._order_cache[target_node]
        return {edge: float(len(ordered) - rank) for rank, edge in enumerate(ordered)}

    def attack(
        self, target_node: int, budget: int, add_edge: bool = False
    ):
        cached = self._order_cache.get(target_node)
        if cached is None or len(cached) < budget:
            max_b = max(budget, getattr(self.system_config.general, "budget", 5))
            self._order_cache[target_node] = self._compute_attack_order(
                target_node, max_b
            )

        ordered = self._order_cache[target_node]
        edges = ordered[:budget]

        embedding_list = []
        for _ in range(self.trials):
            if not edges:
                embedding_list.append(self.data.clone())
                continue
            embedding_list.append(self._remove_edges(edges))
        return embedding_list



class NAGRAttackRetriever(NAGRAttack):
    """NAG-R with embedding-drift loss instead of classification CE."""

    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        if retriever is None:
            raise ValueError("NAGRAttack requires retriever")

        super(NAGRAttack, self).__init__(cfg, system_config, data)

        self.decay = float(getattr(cfg, "decay", 0.9))
        self.step = float(getattr(cfg, "step", 0.7))
        self.embedding_loss_type = getattr(cfg, "embedding_loss_type", "cosine")
        # "query" (drift from the centroid query the metric ranks against) | "self" (drift from the
        # target's OWN clean embedding — query-free ablation; undirected movement, not aimed at the query).
        self.drift_reference = getattr(cfg, "drift_reference", "query")
        self.sort_reverse = True

        cache_dir = getattr(system_config.general, "cache_dir", "cache")
        data_name = system_config.dataset.name
        retriever_name = system_config.retriever.name
        graph_model = getattr(system_config.general, "graph_model", "gcn")

        self.retriever_surrogate_time = 0.0
        self._surrogate_runs = []
        for run_idx in range(self.trials):
            run_id = f"run{run_idx + 1}" if self.trials > 1 else None
            surrogate, clean_embeddings = get_or_train_surrogate_retriever(
                data,
                retriever,
                data_name,
                retriever_name,
                graph_model=graph_model,
                cache_dir=cache_dir,
                hidden_channels=getattr(cfg, "surrogate_hidden", 256),
                epochs=getattr(cfg, "surrogate_epochs", 200),
                lr=getattr(cfg, "surrogate_lr", 0.01),
                weight_decay=getattr(cfg, "surrogate_weight_decay", 5e-4),
                dropout=getattr(cfg, "surrogate_dropout", 0.0),
                loss_type=getattr(cfg, "surrogate_loss_type", "cosine"),
                patience=getattr(cfg, "surrogate_patience", 30),
                device=str(self.device),
                run_id=run_id,
                seed=self.seed + run_idx,
            )
            self.retriever_surrogate_time += float(getattr(surrogate, "retriever_surrogate_time", 0.0))
            surrogate.eval()
            for p in surrogate.parameters():
                p.requires_grad_(False)
            self._surrogate_runs.append((surrogate, clean_embeddings.to(self.device)))
        self._order_cache_by_run = [{} for _ in range(self.trials)]
        self._query_cache = {}             # (run_idx, target) -> frozen centroid query (the metric's query)
        self._clean_emb_full_cache = {}    # run_idx -> surrogate clean all-node embeddings
        self._activate_surrogate_run(0)

        n = data.num_nodes
        self._n = n
        ei = data.edge_index.to(self.device)
        no_self = ei[0] != ei[1]
        self._edge_index = ei[:, no_self]
        self._E = self._edge_index.shape[1]

        self._x = data.x.to(self.device)
        self._xW1 = (self._x @ self._W1.t()).detach()

        self_idx = torch.arange(n, device=self.device)
        self._self_loop_indices = torch.stack([self_idx, self_idx])
        self._order_cache = self._order_cache_by_run[0]

        logger.info(
            f"NAG-R-Retriever initialized: loss={self.embedding_loss_type}, "
            f"decay={self.decay}, step={self.step}, E={self._E}, N={n}"
        )


    def _activate_surrogate_run(self, run_idx: int):
        self._active_surrogate_run = int(run_idx)
        self.surrogate, self._clean_embeddings = self._surrogate_runs[run_idx]
        self._W1 = self.surrogate.W1.weight.detach()
        self._b1 = self.surrogate.W1.bias.detach()
        self._W2 = self.surrogate.W2.weight.detach()
        self._b2 = self.surrogate.W2.bias.detach()
        self._order_cache = self._order_cache_by_run[run_idx]
        if hasattr(self, "_x"):
            self._xW1 = (self._x @ self._W1.t()).detach()

    def attack(self, target_node: int, budget: int, add_edge: bool = False):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        embedding_list = []
        max_b = max(budget, getattr(self.system_config.general, "budget", 5))
        for run_idx in range(self.trials):
            self._activate_surrogate_run(run_idx)
            cached = self._order_cache.get(target_node)
            if cached is None or len(cached) < budget:
                self._order_cache[target_node] = self._compute_attack_order(target_node, max_b)
            edges = self._order_cache[target_node][:budget]
            embedding_list.append(self._remove_edges(edges) if edges else self.data.clone())
        self._activate_surrogate_run(0)
        return embedding_list

    def _centroid_query(self, target_node):
        """Frozen NEIGHBORHOOD-CENTROID query in the surrogate's embedding space — the SAME query
        family the evaluator ranks against (src/oracle/query.py, per-dataset query_radius), built
        once per surrogate run on the CLEAN graph so it does not move under attack. Maximizing
        1 - cos(current, q) pushes the attacked embedding away from the centroid the metric scores
        against, i.e. DEMOTES. (Earlier ports drifted from the target's isolated/full-graph
        embedding — a DIFFERENT reference than the metric, so they optimized the wrong objective.)
        The clean all-node embeddings use a zero look-ahead delta, i.e. the plain forward."""
        key = (self._active_surrogate_run, target_node)
        q = self._query_cache.get(key)
        if q is None:
            run = self._active_surrogate_run
            zc = self._clean_emb_full_cache.get(run)
            if zc is None:
                zeros = torch.zeros(self._n, device=self.device)
                with torch.no_grad():
                    zc = self._gcn_forward_with_lookahead(
                        torch.ones(self._E, device=self.device), 0, zeros).detach()
                self._clean_emb_full_cache[run] = zc
            if self.drift_reference == "self":
                q = zc[target_node:target_node + 1].detach()   # SELF-DRIFT: target's own clean emb (query-free)
            else:
                q = neighborhood_centroid_query(
                    self._edge_index, target_node,
                    self.system_config.dataset.query_radius, zc, self._n,
                ).detach()
            self._query_cache[key] = q
        return q

    def _target_embedding_loss(self, out, target_node):
        current = out[target_node : target_node + 1]
        clean = self._centroid_query(target_node)
        if self.embedding_loss_type == "mse":
            return F.mse_loss(current, clean)
        return 1.0 - F.cosine_similarity(current, clean, dim=1).mean()

    def _compute_target_grad(
        self,
        target_node: int,
        edge_values_current: torch.Tensor,
        momentum_target_row: torch.Tensor,
    ) -> torch.Tensor:
        edge_values = edge_values_current.detach().clone().requires_grad_(True)
        delta_target_row = (self.decay * self.step * momentum_target_row).detach()
        out = self._gcn_forward_with_lookahead(edge_values, target_node, delta_target_row)
        loss = self._target_embedding_loss(out, target_node)
        (grad,) = torch.autograd.grad(loss, edge_values)
        return grad.detach()
