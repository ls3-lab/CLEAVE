"""AGPOA variant using a retriever-embedding surrogate."""
"""
AGPOA: Average Gradient and Perturbation Optimization Attack

Faithful implementation of:
"Enhanced targeted attacks on Graph Neural Networks via Average Gradient
and Perturbation Optimization" (Chen et al., Engineering Applications of AI, 2025)

Module 1 - Average Gradient: accumulates gradient information across all
    attack iterations (Eqs 6-11) to escape local optima that single-step
    or momentum methods fall into.
Module 2 - Perturbation Optimization: uses feature similarity (Eq 15) and
    node homogeneity (Eq 16) with Top-K overlapping (Eq 17) to select
    perturbation edges within the budget constraint (Eq 18).

Adapted for the edge-removal attack framework: uses a surrogate 2-layer GCN
for gradient computation (frozen weights, gradients flow through adjacency).

GPU memory: uses sparse adjacency and torch.sparse.mm to avoid O(N^2) dense
matrices. Memory is O(E) instead of O(N^2), enabling large graphs like
Coauthor-Physics (N=34k) that would otherwise OOM.

Class logits (not embeddings) — to compute softmax predictions
Node labels (data.y) — to compute cross-entropy loss
Backprop through a classification model
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data
from typing import Dict, List, Tuple
from src.attackers.base import Attacker
from src.attackers.surrogate_gcn import get_or_train_surrogate_gcn
from src.attackers.surrogate_retriever import get_or_train_surrogate_retriever
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)



class AGPOAAttack(Attacker):

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        super().__init__(cfg, system_config, data)

        self.momentum_factor = getattr(cfg, 'momentum_factor', 0.6)
        self.budget_alpha = getattr(cfg, 'budget_alpha', 1.0)
        self.num_gradient_modifications = getattr(cfg, 'num_gradient_modifications', 1000)
        self.top_k = getattr(cfg, 'top_k', 50)
        self.sort_reverse = True

        cache_dir = getattr(system_config.general, 'cache_dir', 'cache')
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

        self._order_cache: Dict[int, List[Tuple[int, int]]] = {}

        logger.info(
            f"AGPOA initialized (sparse): \u03bc={self.momentum_factor}, \u03b1={self.budget_alpha}, "
            f"T={self.num_gradient_modifications}, top_k={self.top_k}, "
            f"E={self._E}, N={n}"
        )

    @staticmethod
    def _extract_weight(conv):
        if hasattr(conv, 'lin'):
            return conv.lin.weight.detach()
        return conv.weight.detach()

    @staticmethod
    def _extract_bias(conv):
        if hasattr(conv, 'bias') and conv.bias is not None:
            return conv.bias.detach()
        return None

    def _gcn_forward_sparse(self, edge_values):
        """Sparse 2-layer GCN forward, differentiable w.r.t. edge_values.

        Replicates Eq 1-2: Z = softmax(A_norm ReLU(A_norm X W1) W2)
        using sparse matmul. Gradient flows through both edge values and
        degree normalization via autograd.
        """
        n = self._n

        all_indices = torch.cat([self._edge_index, self._self_loop_indices], dim=1)
        all_values = torch.cat([edge_values, torch.ones(n, device=self.device)])

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

        h = torch.sparse.mm(adj_norm, self._xW1)
        if self._b1 is not None:
            h = h + self._b1
        h = F.relu(h)

        out = torch.sparse.mm(adj_norm, h @ self._W2.t())
        if self._b2 is not None:
            out = out + self._b2

        return out

    def _compute_adj_gradient(self, target_node, edge_values_current):
        """Compute dL/d(edge_values) for the target node (Eq 6)."""
        edge_values = edge_values_current.detach().clone().requires_grad_(True)
        out = self._gcn_forward_sparse(edge_values)
        loss = F.cross_entropy(
            out[target_node : target_node + 1],
            self._labels[target_node : target_node + 1],
        )
        loss.backward()
        return edge_values.grad.detach()

    # ------------------------------------------------------------------
    # Module 1: Average Gradient Calculation  (Algorithm 1, lines 3-9)
    # ------------------------------------------------------------------

    def _average_gradient_module(self, target_node, max_perturbations):
        """Greedy edge removal guided by accumulated average gradient.

        At each step t:
          1. Compute gradient B^(t) on current adjacency        (Eq 6)
          2. Update B_bar^(t+1) = mu B_bar^(t) + B^(t)/||B^(t)||_1   (Eq 9)
          3. Select best removal edge via argmax |B_bar|            (Eq 10)
          4. Flip it in the working adjacency                    (Eq 11)

        Sums both gradient directions per neighbor for undirected graphs.
        Returns ordered list of (target_node, neighbor) removal edges.
        """
        edge_values = torch.ones(self._E, device=self.device)
        avg_grad = torch.zeros(self._E, device=self.device)
        perturbations: List[Tuple[int, int]] = []

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

        removed = set()

        for t in range(self.num_gradient_modifications):
            grad = self._compute_adj_gradient(target_node, edge_values)

            grad_l1 = grad.abs().sum()
            if grad_l1 > 0:
                avg_grad = self.momentum_factor * avg_grad + grad / grad_l1

            best_score = float('-inf')
            best_j = -1
            best_edge = None

            for j in set(nb_to_out) | set(nb_to_in):
                if j in removed:
                    continue
                out_grad = avg_grad[nb_to_out[j]].item() if j in nb_to_out else 0.0
                in_grad = avg_grad[nb_to_in[j]].item() if j in nb_to_in else 0.0
                score = -out_grad - in_grad
                if score > best_score:
                    best_score = score
                    best_j = j
                    best_edge = (j, target_node) if in_grad <= out_grad and j in nb_to_in else (target_node, j)

            if best_j == -1 or best_score <= 0:
                break

            if best_j in nb_to_out:
                edge_values[nb_to_out[best_j]] = 0
            if best_j in nb_to_in:
                edge_values[nb_to_in[best_j]] = 0

            removed.add(best_j)
            perturbations.append(best_edge)

            if len(perturbations) >= max_perturbations:
                break

        return perturbations

    # ------------------------------------------------------------------
    # Module 2: Perturbation Optimization  (Algorithm 1, lines 10-14)
    # ------------------------------------------------------------------

    def _perturbation_optimization(self, target_node, gradient_edges, max_budget):
        """Fill remaining budget using feature similarity and homogeneity.

        Fs[i] = ||X_i - X_tar||  (Eq 15)  -- lower = more similar
        Ho[i] = cos(X_i, X_tar)  (Eq 16)  -- higher = more homogeneous
        Top-K overlapping         (Eq 17)  -- intersection of smallest |Fs|
                                             and largest Ho
        """
        # Orientation-independent: Module 1 may store an edge as (neighbor, target) (agpoa.py ~209),
        # so {j for (_, j) in ...} would grab `target`, not the neighbor, and fail to exclude it.
        already = {self.get_edge_neighbor(e, target_node) for e in gradient_edges}
        remaining = max_budget - len(gradient_edges)
        if remaining <= 0:
            return []

        x_tar = self._x[target_node]
        fs = torch.norm(self._x - x_tar, dim=1)

        x_norm = F.normalize(self._x, p=2, dim=1)
        x_tar_norm = F.normalize(x_tar.unsqueeze(0), p=2, dim=1)
        ho = (x_norm * x_tar_norm).sum(dim=1)

        k = min(self.top_k, self._n - 1)

        fs_tmp = fs.clone()
        fs_tmp[target_node] = float('inf')
        _, fs_topk = torch.topk(-fs_tmp, k)

        ho_tmp = ho.clone()
        ho_tmp[target_node] = float('-inf')
        _, ho_topk = torch.topk(ho_tmp, k)

        similar = set(fs_topk.cpu().tolist()) & set(ho_topk.cpu().tolist())
        if not similar:
            similar = set(fs_topk.cpu().tolist()) | set(ho_topk.cpu().tolist())

        candidate_edges = self.get_target_edges(target_node)
        candidates = [
            (edge, self.get_edge_neighbor(edge, target_node), ho[self.get_edge_neighbor(edge, target_node)].item())
            for edge in candidate_edges
            if self.get_edge_neighbor(edge, target_node) not in already
            and self.get_edge_neighbor(edge, target_node) in similar
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)

        if len(candidates) < remaining:
            used = already | {nb for _, nb, _ in candidates}
            fallback = [
                (edge, self.get_edge_neighbor(edge, target_node), ho[self.get_edge_neighbor(edge, target_node)].item())
                for edge in candidate_edges
                if self.get_edge_neighbor(edge, target_node) not in used
            ]
            fallback.sort(key=lambda x: x[2], reverse=True)
            candidates.extend(fallback)

        return [edge for edge, _, _ in candidates[:remaining]]

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def _compute_attack_order(self, target_node, max_budget):
        """Run Module 1 + Module 2, return ordered edge-removal list."""
        gradient_edges = self._average_gradient_module(target_node, max_budget)

        if len(gradient_edges) < max_budget:
            extra = self._perturbation_optimization(
                target_node, gradient_edges, max_budget
            )
            gradient_edges = gradient_edges + extra

        # Dedup by neighbor (orientation-independent): budget must count UNIQUE undirected edges even
        # if Module 1/2 selected the same neighbor in different tuple orders (else _remove_edges
        # collapses the duplicate and the effective budget shrinks below the requested value).
        seen, deduped = set(), []
        for e in gradient_edges:
            nb = self.get_edge_neighbor(e, target_node)
            if nb not in seen:
                seen.add(nb)
                deduped.append(e)
        return deduped[:max_budget]

    def score_edges(self, target_node):
        if target_node not in self._order_cache:
            max_b = getattr(self.system_config.general, 'budget', 5)
            self._order_cache[target_node] = self._compute_attack_order(
                target_node, max_b
            )

        ordered = self._order_cache[target_node]
        return {edge: len(ordered) - rank for rank, edge in enumerate(ordered)}

    def attack(self, target_node, budget, add_edge=False):
        if target_node not in self._order_cache:
            max_b = max(budget, getattr(self.system_config.general, 'budget', 5))
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


class AGPOAAttackRetriever(AGPOAAttack):
    """AGPOA with embedding-drift loss instead of classification CE."""

    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        if retriever is None:
            raise ValueError("AGPOAAttackRetriever requires retriever")

        super(AGPOAAttack, self).__init__(cfg, system_config, data)

        self.momentum_factor = getattr(cfg, "momentum_factor", 0.6)
        self.budget_alpha = getattr(cfg, "budget_alpha", 1.0)
        self.num_gradient_modifications = getattr(cfg, "num_gradient_modifications", 1000)
        self.top_k = getattr(cfg, "top_k", 50)
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
            f"AGPOA-Retriever initialized: loss={self.embedding_loss_type}, "
            f"T={self.num_gradient_modifications}, top_k={self.top_k}, E={self._E}, N={n}"
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

    def attack(self, target_node, budget, add_edge=False):
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
        embedding — a DIFFERENT reference than the metric, so they optimized the wrong objective.)"""
        key = (self._active_surrogate_run, target_node)
        q = self._query_cache.get(key)
        if q is None:
            run = self._active_surrogate_run
            zc = self._clean_emb_full_cache.get(run)
            if zc is None:
                with torch.no_grad():
                    zc = self._gcn_forward_sparse(torch.ones(self._E, device=self.device)).detach()
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

    def _compute_adj_gradient(self, target_node, edge_values_current):
        edge_values = edge_values_current.detach().clone().requires_grad_(True)
        out = self._gcn_forward_sparse(edge_values)
        loss = self._target_embedding_loss(out, target_node)
        loss.backward()
        return edge_values.grad.detach()
