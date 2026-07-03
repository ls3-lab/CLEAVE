"""InfAttack baseline adapted from Wang et al. (WSDM 2024).

Original: https://github.com/ventr1c/InfAttack. Adaptations for this
retrieval-setting baseline: (1) removal-only — the paper's add variant does
not apply here; (2) any passed-in retriever is ignored — pred_Y is the LR
surrogate's class probabilities, as in OTA/main.py; (3) greedy commit-and-
rescore is preserved in both modes; (4) removal candidates restricted to
neighbors with label == yv, matching OTA's candidate_nodes_set.
"""

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.data.splits import load_or_create_attack_split


class InfAttack(Attacker):
    def __init__(self, cfg, system_config: DictConfig, data: Data,
                 retriever=None, oracle_retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)

        self.K = int(cfg.K) if hasattr(cfg, 'K') else 2
        self.attack_mode = cfg.attack_mode if hasattr(cfg, 'attack_mode') else 'one-time'
        if self.attack_mode not in ('one-time', 'iterative'):
            raise ValueError(f"attack_mode must be 'one-time' or 'iterative', got {self.attack_mode!r}")
        if not hasattr(data, 'y') or data.y is None:
            raise ValueError("InfAttack requires data.y (node labels) to fit the LR surrogate.")

        # Threat model (faithful to Wang et al., WSDM 2024, "restricted black-box"): the LR feature
        # surrogate that produces pred_Y is trained ONLY on the attack-split TRAIN ids, which are
        # DISJOINT from the test targets the harness attacks/evaluates — so no target's label leaks
        # into the surrogate (the original trains on idx_train and attacks idx_test). label_mode
        # selects the paper's two label settings:
        #   'kl' (Known Labels)   -> ground-truth target label yv + same-GT-label candidate filtering
        #   'ul' (Unknown Labels) -> the surrogate's PREDICTED labels instead (no GT for targets)
        self.label_mode = getattr(cfg, "label_mode", "kl")
        if self.label_mode not in ("kl", "ul"):
            raise ValueError(f"label_mode must be 'kl' or 'ul', got {self.label_mode!r}")
        train_ids, _ = load_or_create_attack_split(
            data, system_config.dataset.name,
            getattr(system_config.general, "cache_dir", "cache"),
            seed=getattr(system_config.general, "seed", 42),
        )
        self._train_ids = np.asarray(train_ids, dtype=int)
        self._gt_labels = self.data.y.cpu().numpy()

        self._precompute_adjacency()
        self.pred_Y = self._fit_lr(self._train_ids)
        self.pred_label = np.argsort(self.pred_Y, axis=1)[:, ::-1]
        # Labels used for the target class yv and same-label candidate filtering: ground-truth in
        # KL; the surrogate's argmax prediction in UL (no ground-truth access for the targets).
        self._labels_np = self._gt_labels if self.label_mode == "kl" \
            else np.argmax(self.pred_Y, axis=1).astype(self._gt_labels.dtype)

        self._order_cache: Dict[int, List[Tuple[int, int]]] = {}

    def attack(self, target_node, budget: int) -> list:
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
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())

        self._reset_graph_state()
        yv = self._get_node_label(target_node)
        c = self._get_target_class(target_node, yv)
        candidate_edges = self._candidate_remove_edges(target_node, yv)
        if not candidate_edges:
            return {}
        CN_r_arr = np.array(
            [self.get_edge_neighbor(edge, target_node) for edge in candidate_edges],
            dtype=int,
        )
        delta = self._compute_delta_remove(target_node, CN_r_arr, yv, c, self.K)
        C_0 = self._label_influence_remove(target_node, yv, c, self.K)
        I_l = C_0 + delta
        return {candidate_edges[j]: -float(I_l[j]) for j in range(len(candidate_edges))}

    def _compute_greedy_order(self, target_node: int, budget: int) -> List[Tuple[int, int]]:
        if self.attack_mode == 'one-time':
            return self._one_time_order(target_node, budget)
        return self._iterative_order(target_node, budget)

    def _one_time_order(self, target_node: int, budget: int) -> List[Tuple[int, int]]:
        self._reset_graph_state()
        yv = self._get_node_label(target_node)
        c = self._get_target_class(target_node, yv)

        candidate_edges = self._candidate_remove_edges(target_node, yv)
        if not candidate_edges:
            return []
        CN_r_arr = np.array(
            [self.get_edge_neighbor(edge, target_node) for edge in candidate_edges],
            dtype=int,
        )

        # Deltas computed once (OTA/attack.py:94); only C_0 is refreshed per step.
        delta_I_l_r = self._compute_delta_remove(target_node, CN_r_arr, yv, c, self.K)
        mask = np.ones(len(CN_r_arr), dtype=bool)

        ordered: List[Tuple[int, int]] = []
        for _ in range(budget):
            if self.d[target_node] <= 1 or not mask.any():
                break
            C_0_r = self._label_influence_remove(target_node, yv, c, self.K)
            I_l = C_0_r + delta_I_l_r
            I_l[~mask] = -np.inf
            idx = int(np.argmax(I_l))
            picked = int(CN_r_arr[idx])
            picked_edge = candidate_edges[idx]
            mask[idx] = False

            self.A_hat[target_node, picked] = 0
            self.A_hat[picked, target_node] = 0
            self.d[target_node] -= 1
            ordered.append(picked_edge)

        return ordered

    def _iterative_order(self, target_node: int, budget: int) -> List[Tuple[int, int]]:
        self._reset_graph_state()
        yv = self._get_node_label(target_node)
        c = self._get_target_class(target_node, yv)

        ordered: List[Tuple[int, int]] = []
        for _ in range(budget):
            if self.d[target_node] <= 1:
                break
            candidate_edges = self._candidate_remove_edges(target_node, yv)
            if not candidate_edges:
                break
            CN_r_arr = np.array(
                [self.get_edge_neighbor(edge, target_node) for edge in candidate_edges],
                dtype=int,
            )
            I_l = self._iter_label_influence_remove(target_node, CN_r_arr, yv, c, self.K)
            idx = int(np.argmax(I_l))
            picked = int(CN_r_arr[idx])
            picked_edge = candidate_edges[idx]

            self.A_hat[target_node, picked] = 0
            self.A_hat[picked, target_node] = 0
            self.d[target_node] -= 1
            ordered.append(picked_edge)

        return ordered

    def _original_label_influence(self, K: int, v: int, s: float, yv: int, c: int) -> None:
        if K == 0:
            self.I_l_yv += s * self.pred_Y[v, yv]
            self.I_l_c += s * self.pred_Y[v, c]
            return
        for n in self.A_hat[v].nonzero()[1]:
            w = (1 / self.d[v] ** 0.5) * (1 / self.d[n] ** 0.5)
            self._original_label_influence(K - 1, int(n), s * w, yv, c)

    def _compute_delta_remove(self, target_node: int, CN_r: np.ndarray,
                              yv: int, c: int, K: int) -> np.ndarray:
        delta = np.empty(len(CN_r))
        d_target_orig = self.d[target_node]
        self.d[target_node] = d_target_orig - 1
        for j, u in enumerate(CN_r):
            u = int(u)
            self.I_l_c, self.I_l_yv = 0, 0
            self._original_label_influence(K - 1, u, 1, yv, c)
            val = (1 / self.d[u] ** 0.5) * (1 / self.d[target_node] ** 0.5) \
                  * (self.I_l_yv - self.I_l_c)
            if K == 2:
                val += (1 / self.d[u] ** 0.5) * (1 / self.d[target_node] ** 1.5) \
                       * (self.pred_Y[u, yv] - self.pred_Y[u, c])
            delta[j] = val
        self.d[target_node] = d_target_orig
        return delta

    def _label_influence_remove(self, target_node: int, yv: int, c: int, K: int) -> float:
        d_target_orig = self.d[target_node]
        self.d[target_node] = d_target_orig - 1
        self.I_l_c, self.I_l_yv = 0, 0
        self._original_label_influence(K, target_node, 1, yv, c)
        C_0 = self.I_l_c - self.I_l_yv
        self.d[target_node] = d_target_orig
        return C_0

    def _iter_label_influence_remove(self, target_node: int, CN_r: np.ndarray,
                                     yv: int, c: int, K: int) -> np.ndarray:
        I_l = np.empty(len(CN_r))
        d_target_orig = self.d[target_node]
        self.d[target_node] = d_target_orig - 1
        for j, u in enumerate(CN_r):
            u = int(u)
            self.I_l_c, self.I_l_yv = 0, 0
            d_u_orig = self.d[u]
            self.d[u] = d_u_orig - 1
            self.A_hat[target_node, u] = 0
            self.A_hat[u, target_node] = 0
            self._original_label_influence(K, target_node, 1, yv, c)
            I_l[j] = self.I_l_c - self.I_l_yv
            self.A_hat[target_node, u] = 1
            self.A_hat[u, target_node] = 1
            self.d[u] = d_u_orig
        self.d[target_node] = d_target_orig
        return I_l

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
        self.A_ori = adj
        self.A_hat = adj.copy()
        self.d = np.squeeze(np.array(self.A_ori.sum(1)), 1)

        x_np = self.data.x.cpu().numpy() if isinstance(self.data.x, torch.Tensor) else self.data.x.numpy()
        self.X = sp.csr_matrix(x_np)

    def _reset_graph_state(self) -> None:
        self.A_hat = self.A_ori.copy()
        self.d = np.squeeze(np.array(self.A_ori.sum(1)), 1).copy()

    def _fit_lr(self, train_ids: np.ndarray) -> np.ndarray:
        # LR surrogate trained on the attack-split TRAIN ids ONLY (disjoint from the test targets),
        # always on ground-truth TRAIN labels (training labels are known in the threat model; this
        # is leakage-free precisely because train ∩ test = ∅). Produces probabilities for ALL nodes.
        features_np = self.X.toarray()
        lr = LogisticRegression(
            C=1, penalty='l2', solver='lbfgs',
            fit_intercept=True, max_iter=100,
        )
        lr.fit(features_np[train_ids], self._gt_labels[train_ids])
        probs = lr.predict_proba(features_np)

        # Realign columns to full class space so indexing by class id is safe.
        num_classes = int(self._gt_labels.max()) + 1
        aligned = np.zeros((features_np.shape[0], num_classes), dtype=probs.dtype)
        for idx, cls in enumerate(lr.classes_):
            aligned[:, int(cls)] = probs[:, idx]
        return aligned

    def _candidate_remove_nodes(self, target_node: int, yv: int) -> List[int]:
        return [int(n) for n in self.A_hat[target_node].nonzero()[1]
                if int(n) != target_node and int(self._labels_np[n]) == yv]

    def _candidate_remove_edges(self, target_node: int, yv: int) -> List[Tuple[int, int]]:
        candidates = []
        for edge in self.get_target_edges(target_node):
            neighbor = self.get_edge_neighbor(edge, target_node)
            if neighbor == target_node:
                continue
            if self.A_hat[target_node, neighbor] == 0:
                continue
            if int(self._labels_np[neighbor]) == yv:
                candidates.append(edge)
        return candidates

    def _get_node_label(self, target_node: int) -> int:
        return int(self._labels_np[target_node])

    def _get_target_class(self, target_node: int, yv: int) -> int:
        # Second-best LR class, skipping yv so the attack objective is non-degenerate.
        for cls in self.pred_label[target_node]:
            if int(cls) != yv:
                return int(cls)
        return int(self.pred_label[target_node, 0])
