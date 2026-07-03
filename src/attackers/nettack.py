"""Nettack-inspired variant using a retriever-embedding surrogate.

The original Nettack structural score is a closed-form class-margin objective.
That formula assumes class logits and cannot be reused unchanged for retrieval
embeddings. This variant keeps the greedy targeted edge-removal shape, but
scores each candidate removal by the target embedding drift under a
retriever-trained surrogate GCN.
"""


"""
Nettack attack implementation (GPU-accelerated).
Faithful replication of "Adversarial Attacks on Neural Networks for Graph Data"
by Daniel Zuegner, Amir Akbarnejad and Stephan Guennemann (KDD 2018).

Reference: https://github.com/danielzuegner/nettack

GPU acceleration: uses dense adjacency on GPU with analytical batch computation
of structure scores via the decomposition:

    A_hat^2[u,v] = d^{-1/2}_u * d^{-1/2}_v * sum_j (A+I)[u,j] * d^{-1}_j * (A+I)[j,v]

The sum is a single vector-matrix multiply (O(N^2)) computed once per greedy
step. Per-candidate edge-removal corrections are O(N) rank-1 updates, and
struct scores for all K candidates are computed in a single batched matmul.

Class logits (not embeddings) — to compute softmax predictions
Node labels (data.y) — to compute cross-entropy loss
Backprop through a classification model
"""

import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data
from  src.attackers.surrogate_gcn import get_or_train_surrogate_gcn
from src.attackers.base import Attacker
from src.attackers.surrogate_retriever import get_or_train_surrogate_retriever
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

def compute_alpha(n, S_d, d_min):
    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def compute_log_likelihood(n, alpha, S_d, d_min):
    return n * np.log(alpha) + n * alpha * np.log(d_min) - (alpha + 1) * S_d


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)
    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)
    return new_S_d, new_n


def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff


class NettackGPU:
    """GPU-accelerated Nettack core using dense adjacency."""

    def __init__(self, adj, x, labels, W, u, device):
        """
        Parameters
        ----------
        adj : torch.Tensor
            N×N dense adjacency (no self-loops) on *device*.
        x : torch.Tensor
            N×F feature matrix on *device*.
        labels : torch.Tensor
            N label vector on *device*.
        W : torch.Tensor
            F×C combined weight matrix (W1^T @ W2^T) on *device*.
        u : int
            Target node index.
        device : str or torch.device
        """
        self.device = device
        self.N = adj.shape[0]
        self.u = u
        self.label_u = int(labels[u])
        self.K_classes = int(labels.max()) + 1

        self.adj = adj
        self.deg = adj.sum(1) + 1  # A+I degree
        self.XW = x @ W

        self.neighbors_set = set(adj[u].nonzero(as_tuple=True)[0].tolist())
        self.structure_perturbations: List[Tuple[int, int]] = []

    def compute_logits(self):
        """A_hat^2[u,:] @ XW without materializing A_hat."""
        d_inv = self.deg.pow(-1.0)
        d_inv[self.deg <= 0] = 0
        d_inv_sqrt = self.deg.pow(-0.5)
        d_inv_sqrt[self.deg <= 0] = 0

        row_u = self.adj[self.u].clone()
        row_u[self.u] += 1
        c = row_u * d_inv
        temp = c @ self.adj + c  # c @ (A+I)
        a_hat_sq_u = d_inv_sqrt[self.u] * d_inv_sqrt * temp
        return (a_hat_sq_u @ self.XW).cpu().numpy()

    def strongest_wrong_class(self, logits):
        label_onehot = np.eye(self.K_classes)[self.label_u]
        return int((logits - 1000 * label_onehot).argmax())

    def struct_scores_batch(self, candidate_ms):
        """Batch-compute struct scores for removing edges (u, m_i).

        Derivation (removal of edge (u, m) from undirected graph):

          Let d' = d with d'[u]-=1, d'[m]-=1.  Define:
            c = (A+I)[u,:] * d'^{-1}          (shared, u already decremented)
            base = c @ (A+I)                   (computed ONCE)

          Per candidate m_i:
            S_i  = base  -  (1/d[m_i]) * (A+I)[m_i,:]
            S_i[m_i] -= d'^{-1}[u]

            A_hat'^2[u,:] = d'^{-1/2}[u] * d'^{-1/2} * S_i
            (then correct column m_i for the decremented d'[m_i])

        Returns numpy array of shape (K,) — lower = more damaging.
        """
        K = len(candidate_ms)
        if K == 0:
            return np.array([])

        ms = torch.tensor(candidate_ms, device=self.device, dtype=torch.long)
        u = self.u
        d = self.deg
        idx = torch.arange(K, device=self.device)

        # --- shared: degrees with u decremented ---
        d_u_dec = d.clone()
        d_u_dec[u] -= 1

        d_inv = d_u_dec.pow(-1.0)
        d_inv[d_u_dec <= 0] = 0
        d_inv_sqrt = d_u_dec.pow(-0.5)
        d_inv_sqrt[d_u_dec <= 0] = 0

        row_u = self.adj[u].clone()
        row_u[u] += 1
        c = row_u * d_inv
        base = c @ self.adj + c  # c @ (A+I), O(N^2) done once

        # --- per-candidate corrections (all O(K*N)) ---
        d_m = d[ms]
        coeff = -1.0 / d_m
        coeff[d_m <= 0] = 0

        rows_m = self.adj[ms].clone()
        rows_m[idx, ms] += 1  # add self-loops → (A+I)[m_i,:]

        S = base.unsqueeze(0) + coeff.unsqueeze(1) * rows_m
        S[idx, ms] -= d_inv[u]

        A_hat_sq = d_inv_sqrt[u] * d_inv_sqrt.unsqueeze(0) * S

        # correct column m_i for decremented degree
        d_m_new = d_m - 1
        d_inv_sqrt_m_new = d_m_new.pow(-0.5)
        d_inv_sqrt_m_new[d_m_new <= 0] = 0
        d_inv_sqrt_m_old = d_inv_sqrt[ms]
        ratio = torch.where(
            d_inv_sqrt_m_old > 0,
            d_inv_sqrt_m_new / d_inv_sqrt_m_old,
            d_inv_sqrt_m_new,
        )
        A_hat_sq[idx, ms] *= ratio

        # struct score = logit[correct] - max(logit[wrong])
        logits = A_hat_sq @ self.XW
        label_mask = torch.zeros(self.K_classes, device=self.device)
        label_mask[self.label_u] = -1e9
        wrong_logits = (logits + label_mask).max(dim=1).values
        correct_logits = logits[:, self.label_u]

        return (correct_logits - wrong_logits).cpu().numpy()

    def attack_surrogate(self, n_perturbations, delta_cutoff=0.004, removal_only=True):
        logits_start = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits_start)
        surrogate_losses = [logits_start[self.label_u] - logits_start[best_wrong_class]]

        if removal_only:
            neighbors = sorted(self.neighbors_set)
            if not neighbors:
                return self.structure_perturbations
            potential_edges = np.column_stack(
                (np.tile(self.u, len(neighbors)), np.array(neighbors))
            ).astype("int32")
        else:
            potential_edges = np.column_stack(
                (np.tile(self.u, self.N - 1),
                 np.setdiff1d(np.arange(self.N), self.u))
            ).astype("int32")

        # power-law degree tracking (CPU, small vectors)
        degree_sequence_start = (self.deg - 1).cpu().numpy()  # A degree (no self-loop)
        current_degree_sequence = degree_sequence_start.copy()
        d_min = 2
        S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
        n_start = np.sum(degree_sequence_start >= d_min)
        current_n = np.sum(current_degree_sequence >= d_min)
        alpha_start = compute_alpha(n_start, S_d_start, d_min)
        log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

        for step in range(n_perturbations):
            filtered_edges = potential_edges
            if len(filtered_edges) == 0:
                break

            # power-law check
            fe_t = torch.from_numpy(filtered_edges.astype(np.int64)).to(self.device)
            adj_vals = self.adj[fe_t[:, 0], fe_t[:, 1]].cpu().numpy()
            deltas = 2 * (1 - adj_vals) - 1
            d_edges_old = current_degree_sequence[filtered_edges]
            d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, None]

            new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
            new_alphas = compute_alpha(new_n, new_S_d, d_min)
            new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
            alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
            new_ll_combined = compute_log_likelihood(
                new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min,
            )
            new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)

            powerlaw_filter = filter_chisquare(new_ratios, delta_cutoff)
            filtered_edges_final = filtered_edges[powerlaw_filter]

            if len(filtered_edges_final) == 0:
                filtered_edges_final = filtered_edges
                active_filter = np.ones(len(filtered_edges), dtype=bool)
            else:
                active_filter = powerlaw_filter

            # GPU batch struct scores
            struct_scores = self.struct_scores_batch(filtered_edges_final[:, 1].tolist())

            best_edge_ix = int(struct_scores.argmin())
            best_edge = filtered_edges_final[best_edge_ix]

            # apply perturbation
            i, j = int(best_edge[0]), int(best_edge[1])
            was_edge = self.adj[i, j].item() != 0
            self.adj[i, j] = 1 - self.adj[i, j]
            self.adj[j, i] = 1 - self.adj[j, i]
            delta_deg = -1 if was_edge else 1
            self.deg[i] += delta_deg
            self.deg[j] += delta_deg

            if was_edge:
                self.neighbors_set.discard(j if i == self.u else i)
            else:
                self.neighbors_set.add(j if i == self.u else i)

            self.structure_perturbations.append((i, j))

            if removal_only and was_edge:
                mask = ~((potential_edges[:, 0] == i) & (potential_edges[:, 1] == j))
                potential_edges = potential_edges[mask]

            surrogate_losses.append(struct_scores[best_edge_ix])

            current_S_d = new_S_d[active_filter][best_edge_ix]
            current_n = new_n[active_filter][best_edge_ix]
            current_degree_sequence[best_edge] += deltas[active_filter][best_edge_ix]

        return self.structure_perturbations


class NettackAttack(Attacker):
    """Nettack attacker integrated with the framework (GPU-accelerated)."""

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        super().__init__(cfg, system_config, data)
        self.direct_attack = cfg.direct_attack if hasattr(cfg, 'direct_attack') else True
        self.delta_cutoff = cfg.delta_cutoff if hasattr(cfg, 'delta_cutoff') else 0.004
        self.data_name = system_config.dataset.name if hasattr(system_config, 'dataset') else 'unknown'

        self._init_surrogate(data)

    def _init_surrogate(self, data):
        cache_dir = getattr(self.system_config.general, 'cache_dir', 'cache')
        surrogate = get_or_train_surrogate_gcn(
            data, self.data_name, cache_dir=cache_dir, device=str(self.device),
        )
        surrogate.eval()

        W1 = self._extract_weight(surrogate.conv1)
        W2 = self._extract_weight(surrogate.conv2)
        self.W = W1.T @ W2.T  # (input, output) on device

        n = data.num_nodes
        self.adj_dense = torch.zeros(n, n, device=self.device)
        ei = data.edge_index.to(self.device)
        self.adj_dense[ei[0], ei[1]] = 1.0

        self.x = data.x.to(self.device).float()
        self.labels = data.y.to(self.device)

        logger.info(
            f"Nettack GPU initialized: N={n}, device={self.device}, "
            f"adj memory={n * n * 4 / 1e9:.2f} GB"
        )

    @staticmethod
    def _extract_weight(conv):
        if hasattr(conv, 'lin'):
            return conv.lin.weight.detach()
        return conv.weight.detach()

    def score_edges(self, target_node: int) -> dict[Tuple[int, int], float]:
        candidate_edges = self.get_target_edges(target_node)
        if not candidate_edges:
            return {}

        neighbors = [self.get_edge_neighbor(edge, target_node) for edge in candidate_edges]
        nettack = NettackGPU(
            self.adj_dense, self.x, self.labels,
            self.W, target_node, self.device,
        )
        scores_np = nettack.struct_scores_batch(neighbors)

        return {candidate_edges[i]: float(scores_np[i]) for i in range(len(candidate_edges))}

    def attack(self, target_node: int, budget: int) -> List[Data]:
        embedding_list = []
        for i in range(self.trials):
            torch.manual_seed(self.seed + i)
            torch.cuda.manual_seed(self.seed + i)
            random.seed(self.seed + i)
            np.random.seed(self.seed + i)

            nettack = NettackGPU(
                self.adj_dense.clone(), self.x, self.labels,
                self.W, target_node, self.device,
            )
            perturbations = nettack.attack_surrogate(
                n_perturbations=budget,
                delta_cutoff=self.delta_cutoff,
            )

            raw_edges = set(self.get_target_edges(target_node))
            edges_to_remove = []
            for e in perturbations:
                if self.adj_dense[e[0], e[1]].item() == 1:
                    candidates = [(int(e[0]), int(e[1])), (int(e[1]), int(e[0]))]
                    chosen = next((edge for edge in candidates if edge in raw_edges), None)
                    if chosen is not None:
                        edges_to_remove.append(chosen)

            if edges_to_remove:
                modified_data = self._remove_edges(edges_to_remove)
            else:
                modified_data = self.data.clone()

            embedding_list.append(modified_data)

        return embedding_list


class NettackAttackRetriever(Attacker):
    """Greedy Nettack-style removal scored by retriever-surrogate drift."""

    def __init__(self, cfg: DictConfig, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        if retriever is None:
            raise ValueError("NettackAttackRetriever requires retriever")
        super().__init__(cfg, system_config, data)

        self.embedding_loss_type = getattr(cfg, "embedding_loss_type", "cosine")
        # "query" (drift from the centroid query the metric ranks against) | "self" (drift from the
        # target's OWN clean embedding — query-free ablation; undirected movement, not aimed at the query).
        self.drift_reference = getattr(cfg, "drift_reference", "query")
        self.greedy_recompute = bool(getattr(cfg, "greedy_recompute", True))
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
        self.adj_dense = torch.zeros(n, n, device=self.device)
        ei = data.edge_index.to(self.device)
        self.adj_dense[ei[0], ei[1]] = 1.0
        self._clean_ei = ei                # clean edge_index for the centroid-query k-hop BFS
        self.x = data.x.to(self.device).float()
        self._order_cache = self._order_cache_by_run[0]

        logger.info(
            f"Nettack-Retriever initialized: N={n}, loss={self.embedding_loss_type}, "
            f"adj memory={n * n * 4 / 1e9:.2f} GB"
        )


    def _activate_surrogate_run(self, run_idx: int):
        self._active_surrogate_run = int(run_idx)
        self.surrogate, self.clean_embeddings = self._surrogate_runs[run_idx]
        self._order_cache = self._order_cache_by_run[run_idx]

    def _edge_index_from_adj(self, adj):
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        self_idx = torch.arange(adj.size(0), device=self.device)
        self_loops = torch.stack([self_idx, self_idx])
        return torch.cat([edge_index, self_loops], dim=1)

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
                    zc = self.surrogate(self.x, self._edge_index_from_adj(self.adj_dense)).detach()
                self._clean_emb_full_cache[run] = zc
            if self.drift_reference == "self":
                q = zc[target_node:target_node + 1].detach()   # SELF-DRIFT: target's own clean emb (query-free)
            else:
                q = neighborhood_centroid_query(
                    self._clean_ei, target_node,
                    self.system_config.dataset.query_radius, zc, self._n,
                ).detach()
            self._query_cache[key] = q
        return q

    def _target_embedding_score(self, adj, target_node):
        edge_index = self._edge_index_from_adj(adj)
        with torch.no_grad():
            z = self.surrogate(self.x, edge_index)
        current = z[target_node : target_node + 1]
        clean = self._centroid_query(target_node)
        if self.embedding_loss_type == "mse":
            return float(F.mse_loss(current, clean).item())
        return float((1.0 - F.cosine_similarity(current, clean, dim=1).mean()).item())

    def _score_candidate_removal(self, adj, target_node, neighbor):
        trial = adj.clone()
        trial[target_node, neighbor] = 0.0
        trial[neighbor, target_node] = 0.0
        return self._target_embedding_score(trial, target_node)

    def _compute_attack_order(self, target_node: int, budget: int) -> List[Tuple[int, int]]:
        adj = self.adj_dense.clone()
        ordered: List[Tuple[int, int]] = []

        for _ in range(budget):
            candidate_edges = [
                edge for edge in self.get_target_edges(target_node)
                if adj[target_node, self.get_edge_neighbor(edge, target_node)] > 0
            ]
            if not candidate_edges:
                break

            best_edge = None
            best_nb = None
            best_score = -float("inf")
            for edge in candidate_edges:
                nb = self.get_edge_neighbor(edge, target_node)
                score = self._score_candidate_removal(adj, target_node, nb)
                if score > best_score:
                    best_score = score
                    best_nb = nb
                    best_edge = edge

            if best_edge is None:
                break
            adj[target_node, best_nb] = 0.0
            adj[best_nb, target_node] = 0.0
            ordered.append(best_edge)

            if not self.greedy_recompute:
                remaining = [edge for edge in candidate_edges if self.get_edge_neighbor(edge, target_node) != best_nb]
                scored = [
                    (edge, self._score_candidate_removal(adj, target_node, self.get_edge_neighbor(edge, target_node)))
                    for edge in remaining
                ]
                scored.sort(key=lambda item: item[1], reverse=True)
                ordered.extend(edge for edge, _ in scored[: max(0, budget - len(ordered))])
                break

        return ordered[:budget]

    def score_edges(self, target_node: int):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        if target_node not in self._order_cache:
            max_b = getattr(self.system_config.general, "budget", 5)
            self._order_cache[target_node] = self._compute_attack_order(target_node, max_b)
        ordered = self._order_cache[target_node]
        return {edge: float(len(ordered) - rank) for rank, edge in enumerate(ordered)}

    def attack(self, target_node: int, budget: int) -> List[Data]:
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        embedding_list = []
        max_b = max(budget, getattr(self.system_config.general, "budget", 5))
        for i in range(self.trials):
            torch.manual_seed(self.seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed + i)
            random.seed(self.seed + i)
            np.random.seed(self.seed + i)
            self._activate_surrogate_run(i)
            cached = self._order_cache.get(target_node)
            if cached is None or len(cached) < budget:
                self._order_cache[target_node] = self._compute_attack_order(target_node, max_b)
            edges = self._order_cache[target_node][:budget]
            embedding_list.append(self._remove_edges(edges) if edges else self.data.clone())
        self._activate_surrogate_run(0)
        return embedding_list
