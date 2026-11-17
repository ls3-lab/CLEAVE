"""InfMax baseline adapted from Ma et al. (NeurIPS 2020).

Original: https://github.com/TheaperDeng/GNN-Attack-InfMax. Adaptations
for this retrieval-setting baseline:
  - Removal-only. The paper perturbs node features; this variant instead
    removes edges incident to each target.
  - Per target, candidates are the target's neighbors. Scoring preserves
    InfMax-Unif (alpha-clip) / InfMax-Norm (erf) with greedy submodular
    updates on the `s` accumulator, matching ``New_sort`` / ``New_sort_erf``.
  - Influence is summed over test-set receivers by default, mirroring the
    ``New_sort_sumtest`` / ``New_sort_erf_testsum`` variants. Set
    ``use_test_mask=False`` to sum over all nodes.
  - ``method='sum'`` gives a non-greedy RWCS-style column-sum baseline.
"""

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from scipy.special import erf
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.data.splits import split_data, get_target_nodes


class InfMaxAttack(Attacker):
    def __init__(self, cfg, system_config: DictConfig, data: Data,
                 retriever=None, oracle_retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)

        self.K = int(cfg.steps) if hasattr(cfg, 'steps') else 4
        self.alpha = float(cfg.alpha) if hasattr(cfg, 'alpha') else 0.01
        self.sigma = float(cfg.sigma) if hasattr(cfg, 'sigma') else 0.01
        method = cfg.method if hasattr(cfg, 'method') else 'unif'
        method_aliases = {'uniform': 'unif', 'erf': 'norm'}
        self.method = method_aliases.get(method, method)
        if self.method not in ('unif', 'norm', 'sum'):
            raise ValueError(
                f"method must be 'unif'/'uniform', 'norm'/'erf', or 'sum', got {method!r}"
            )
        self.use_test_mask = bool(cfg.use_test_mask) if hasattr(cfg, 'use_test_mask') else True

        self.train_mask, self.val_mask, self.test_mask = split_data(data)
        self.selected_nodes = torch.tensor(
            get_target_nodes(self.test_mask),
            dtype=torch.long,
            device=self.device,
        )

        self._precompute_adjacency()
        self.M = self._compute_random_walk_matrix()

        if self.use_test_mask:
            self._receivers = np.nonzero(self.test_mask.cpu().numpy())[0]
        else:
            self._receivers = np.arange(self.M.shape[0])
        self._M_recv = self.M[self._receivers, :]  # (|R|, N)

        self._order_cache: Dict[int, List[Tuple[int, int]]] = {}

    def attack(self, target_node, budget: int, add_edge: bool = False) -> list:
        if add_edge:
            return self._attack_add(target_node, budget)
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())

        ordered = self._order_cache.get(target_node)
        if ordered is None or len(ordered) < budget:
            ordered = self._compute_greedy_order(target_node, budget)
            self._order_cache[target_node] = ordered

        edges = ordered[:budget]
        if not edges:
            return [self.data.clone()]
        return [self._remove_edges(edges)]

    def score_edges(self, target_node) -> Dict[Tuple[int, int], float]:
        """One-shot score per edge (no greedy state). Kept for interface compat."""
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())

        candidate_edges = self.get_target_edges(target_node)
        if not candidate_edges:
            return {}
        neighbors_arr = np.array(
            [self.get_edge_neighbor(edge, target_node) for edge in candidate_edges],
            dtype=int,
        )
        M_cols = self._M_recv[:, neighbors_arr]

        if self.method == 'unif':
            per = np.minimum(M_cols, self.alpha).sum(axis=0)
        elif self.method == 'norm':
            per = erf(M_cols / (self.sigma * np.sqrt(2))).sum(axis=0)
        else:
            per = M_cols.sum(axis=0)

        return {candidate_edges[j]: -float(per[j])
                for j in range(len(candidate_edges))}

    def _compute_greedy_order(self, target_node: int, budget: int) -> List[Tuple[int, int]]:
        candidate_edges = self.get_target_edges(target_node)
        if not candidate_edges:
            return []
        neighbors_arr = np.array(
            [self.get_edge_neighbor(edge, target_node) for edge in candidate_edges],
            dtype=int,
        )
        n_cand = len(candidate_edges)
        M_cols = self._M_recv[:, neighbors_arr]  # (|R|, n_cand)

        if self.method == 'sum':
            # RWCS-style: independent column sums, no greedy state.
            per = M_cols.sum(axis=0)
            order = np.argsort(-per)[:budget]
            return [candidate_edges[int(j)] for j in order]

        # Greedy submodular with accumulated s.
        s = np.zeros(len(self._receivers), dtype=M_cols.dtype)
        alive = np.ones(n_cand, dtype=bool)
        ordered: List[Tuple[int, int]] = []

        for _ in range(budget):
            if not alive.any():
                break
            L_vals = s[:, None] + M_cols
            if self.method == 'unif':
                L = np.minimum(L_vals, self.alpha)
            else:
                L = erf(L_vals / (self.sigma * np.sqrt(2)))
            per = L.sum(axis=0)
            per[~alive] = -np.inf
            best = int(np.argmax(per))
            s = s + M_cols[:, best]
            alive[best] = False
            ordered.append(candidate_edges[best])

        return ordered

    def _get_target_neighbors(self, target_node: int) -> List[int]:
        edge_index = self.data.edge_index.cpu().numpy()
        src, dst = edge_index[0], edge_index[1]
        others = np.concatenate([dst[src == target_node], src[dst == target_node]])
        uniq = np.unique(others)
        uniq = uniq[uniq != target_node]
        return uniq.tolist()

    def _precompute_adjacency(self) -> None:
        num_nodes = int(self.data.x.shape[0])
        edge_index = self.data.edge_index.cpu().numpy()
        src, dst = edge_index[0], edge_index[1]
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
        adj = sp.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(num_nodes, num_nodes),
        ).tolil()
        adj.setdiag(1)
        self.adj = adj

        # Row-normalize to a random-walk transition matrix (matches getM).
        adj_dense = adj.toarray().astype(np.float32)
        row_sums = adj_dense.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.Prob = adj_dense / row_sums

    def _compute_random_walk_matrix(self) -> np.ndarray:
        M = self.Prob.copy()
        for _ in range(self.K - 1):
            M = M @ self.Prob
        return M

    def score_candidates(self, target_node: int) -> torch.Tensor:
        """Add-edge mode: attach the target to the highest-influence nodes.

        The random-walk influence matrix ``M`` (``self._M_recv``: rows =
        receivers over the test set, columns = nodes) gives each node's
        influence reach. The per-node score is the receiver-summed influence
        column (the RWCS signal), so the base adds edges to the nodes whose
        influence over the retrieval set is largest. Structure-only, query-free.
        """
        import numpy as _np
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        per = _np.asarray(self._M_recv.sum(axis=0)).ravel()   # [N] influence reach
        return torch.tensor(per, dtype=torch.float, device=self.device)

