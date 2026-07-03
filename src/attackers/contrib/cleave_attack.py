"""CLEAVE: Contrastive-Learned Edge Attack Via Embeddings (query-free black-box transfer).

Runs AGPOA's gradient demotion attack (average-gradient edge selection that maximizes
1 - cos(target_emb, frozen neighbourhood-centroid query) in the surrogate's space, i.e.
pushes the target away from the corpus region the evaluator ranks against), but through
a surrogate trained FROM SCRATCH with self-supervised graph contrastive learning on
PUBLIC graph structure + features only. It NEVER queries the victim retriever — no
retriever.encode, no victim/oracle checkpoint — so it uses strictly less access than
NAG-R/AGPOA (which distill the victim's embeddings). User-confirmed black-box.

The surrogate is victim-INDEPENDENT, so the SAME edge removals are produced for all three
retrievers of a dataset (a pure transfer attack), and cached per dataset.
"""
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.agpoa import AGPOAAttack, AGPOAAttackRetriever
from src.attackers.contrib.edges.contrastive_surrogate import get_or_train_contrastive_surrogate
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class CLEAVEAttack(AGPOAAttackRetriever):
    """AGPOA gradient attack on a from-scratch self-supervised contrastive surrogate."""

    def __init__(self, cfg, system_config: DictConfig, data: Data,
                 retriever=None, oracle_retriever=None, **kwargs):
        # Skip AGPOAAttackRetriever.__init__ (it queries the victim); init the base Attacker.
        super(AGPOAAttack, self).__init__(cfg, system_config, data)

        self.momentum_factor = getattr(cfg, 'momentum_factor', 0.6)
        self.budget_alpha = getattr(cfg, 'budget_alpha', 1.0)
        self.num_gradient_modifications = getattr(cfg, 'num_gradient_modifications', 1000)
        self.top_k = getattr(cfg, 'top_k', 50)
        self.embedding_loss_type = getattr(cfg, 'embedding_loss_type', 'cosine')
        # CLEAVE skips AGPOAAttackRetriever.__init__, so set the demotion target here. "query" = drift
        # from the neighbourhood-centroid query in the surrogate's OWN embedding space (victim-free);
        # this is the CLEAVE design and reproduces the published output_cleave results.
        self.drift_reference = getattr(cfg, 'drift_reference', 'query')
        # attack-only frozen-query radius override (decouples q_t from the EVALUATOR's query_radius)
        _qr = getattr(cfg, 'query_radius', None)
        self.attack_query_radius = _qr if _qr not in (None, "", "none", "None") else None
        self.sort_reverse = True

        cache_dir = getattr(system_config.general, 'cache_dir', 'cache')
        data_name = system_config.dataset.name
        # Ensemble of K self-supervised surrogates (different seeds); gradients are averaged
        # per attack step -> consensus edge selection, a more robust victim transfer. K=1
        # reproduces the single-surrogate CLEAVE exactly.
        self.n_surrogates = int(getattr(cfg, 'n_surrogates', 1))

        # GRAPH-FRACTION ablation (S4): give CLEAVE only a random fraction f of the graph's undirected
        # edges -> surrogate trains AND attack runs on this sparsified graph; self.data stays FULL so
        # _remove_edges + victim eval are on the full graph. f=1.0 reproduces current CLEAVE.
        self.graph_fraction = float(getattr(cfg, 'graph_fraction', 1.0))
        _cache_tag = ''
        _surr_data = data
        self._sparse_ei = None
        if self.graph_fraction < 1.0:
            import copy as _copy
            _ei = data.edge_index.to(self.device)
            _ei = _ei[:, _ei[0] != _ei[1]]                      # drop self-loops
            _src, _dst = _ei[0], _ei[1]
            _und = (_src < _dst).nonzero(as_tuple=True)[0]      # canonical undirected pairs
            _gen = torch.Generator(device='cpu').manual_seed(int(self.seed))
            _perm = torch.randperm(_und.numel(), generator=_gen).to(self.device)
            _k = int(round(self.graph_fraction * _und.numel()))
            _keep = _und[_perm[:_k]]
            _ks, _kd = _src[_keep], _dst[_keep]
            self._sparse_ei = torch.cat([torch.stack([_ks, _kd]), torch.stack([_kd, _ks])], dim=1)
            _surr_data = _copy.copy(data); _surr_data.edge_index = self._sparse_ei
            _cache_tag = '_g%d' % int(round(self.graph_fraction * 100))

        # SURROGATE NODE HOLDOUT (leakage check): train the SSL surrogate on an 80/20 node split's
        # TRAIN subgraph only. Every evaluation TEST target is removed (node + its incident edges)
        # from the surrogate's training graph, so the surrogate NEVER sees the nodes it is used to
        # attack. The attack itself still runs on the FULL graph (self._edge_index stays full).
        # holdout=0 reproduces standard (full-graph, transductive) CLEAVE.
        self.surrogate_holdout = float(getattr(cfg, 'surrogate_holdout', 0.0))
        if self.surrogate_holdout > 0.0:
            import copy as _copy
            from src.data.splits import load_or_create_attack_split as _split
            _N = int(data.num_nodes)
            _traint, _testt = _split(data, data_name, cache_dir, seed=int(self.seed))
            _hold = set(int(t) for t in _testt)                          # always hold out the eval TEST targets
            # SELECTION phase: also exclude the TRAIN (validation) targets so hyperparameter selection
            # by demotion on the train targets is unbiased (surrogate never saw them either).
            self.surrogate_exclude_train = bool(getattr(cfg, 'surrogate_exclude_train', False))
            if self.surrogate_exclude_train:
                _hold |= set(int(t) for t in _traint)
            _n_hold = int(round(self.surrogate_holdout * _N))
            _gen = torch.Generator().manual_seed(int(self.seed))
            _others = torch.tensor([i for i in range(_N) if i not in _hold])
            _perm = torch.randperm(_others.numel(), generator=_gen)
            _need = max(0, _n_hold - len(_hold))
            _hold |= set(_others[_perm[:_need]].tolist())                # fill to the holdout fraction
            _train_mask = torch.ones(_N, dtype=torch.bool)
            _train_mask[torch.tensor(sorted(_hold))] = False
            _tn = _train_mask.nonzero(as_tuple=True)[0]                  # train (kept) node ids
            _ei_cpu = data.edge_index.cpu()
            _em = _train_mask[_ei_cpu[0]] & _train_mask[_ei_cpu[1]]      # edges with BOTH ends kept
            _relabel = torch.full((_N,), -1, dtype=torch.long); _relabel[_tn] = torch.arange(_tn.numel())
            _surr_data = _copy.copy(data)
            _surr_data.x = data.x[_tn]
            _surr_data.edge_index = _relabel[_ei_cpu[:, _em]].to(data.edge_index.device)
            _surr_data.num_nodes = int(_tn.numel())
            _cache_tag = '_ho%d%s' % (int(round(self.surrogate_holdout * 100)),
                                      '_xtr' if self.surrogate_exclude_train else '')
            logger.info('[cleave] surrogate node-holdout=%.2f exclude_train=%s: SSL train subgraph N=%d/%d '
                        '(held out %d nodes incl. all %d test%s targets)'
                        % (self.surrogate_holdout, self.surrogate_exclude_train, _tn.numel(), _N, len(_hold),
                           len(_testt), ('+%d train' % len(_traint)) if self.surrogate_exclude_train else ''))

        self.retriever_surrogate_time = 0.0
        self._surrogate_runs = []
        for k in range(self.n_surrogates):
            model = get_or_train_contrastive_surrogate(
                _surr_data, data_name, cache_dir=cache_dir,
                hidden=getattr(cfg, 'surrogate_hidden', 256),
                out=getattr(cfg, 'surrogate_out', 256),
                epochs=getattr(cfg, 'surrogate_epochs', 300),
                lr=getattr(cfg, 'surrogate_lr', 1e-3),
                drop1=getattr(cfg, 'drop1', 0.3), drop2=getattr(cfg, 'drop2', 0.4),
                fmask1=getattr(cfg, 'fmask1', 0.2), fmask2=getattr(cfg, 'fmask2', 0.3),
                tau=getattr(cfg, 'tau', 0.5), batch=getattr(cfg, 'cl_batch', 2048),
                device=str(self.device), seed=self.seed + k, cache_tag=_cache_tag,
            )
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._surrogate_runs.append((model, None))   # clean_embeddings unused by the attack

        self._order_cache_by_run = [{} for _ in range(max(self.trials, self.n_surrogates))]
        self._query_cache = {}
        self._clean_emb_full_cache = {}

        n = data.num_nodes
        self._n = n
        ei = data.edge_index.to(self.device)
        no_self = ei[0] != ei[1]
        # candidate/attack edges = the sparsified set when graph_fraction<1, else the full graph
        self._edge_index = self._sparse_ei if self._sparse_ei is not None else ei[:, no_self]
        self._E = self._edge_index.shape[1]
        self._x = data.x.to(self.device)
        self_idx = torch.arange(n, device=self.device)
        self._self_loop_indices = torch.stack([self_idx, self_idx])

        # DEGREE-AWARE selection (fixes the dense/degree cells e.g. DBLP): the GCN's degree
        # normalization suppresses high-degree-edge gradients, so plain CLEAVE under-selects the hub
        # edges the victim is most sensitive to. degree_bias>0 reweights Module-1 selection by
        # neighbour degree (beta=0 => identical to plain CLEAVE). Victim-free (uses clean topology only).
        self.degree_bias = float(getattr(cfg, 'degree_bias', 0.0))
        # bias_signal: which training-free importance signal reweights Module-1 selection.
        #   'degree' -> neighbour degree (boosts hubs; fixes DBLP-type cells)
        #   'thp'    -> TwoHopProbing score cos(x_t,x_j)*(1+excl)/sqrt(deg_j) (richer; feature +
        #               2-hop-exclusivity aware, degree-DOWN-weighted) for the cells degree can't fix
        self.bias_signal = str(getattr(cfg, 'bias_signal', 'degree'))
        # degree_adaptive: per target, pick between the biased (degree_bias) and plain (beta=0)
        # removal set by which one drifts the target more in the SURROGATE's own ranking (victim-free).
        # Gets the DBLP-type gains without hurting the gradient cells (which prefer plain).
        self.degree_adaptive = bool(getattr(cfg, 'degree_adaptive', False))
        self.b1_margin = bool(getattr(cfg, 'b1_margin', False))  # RA-IF single-edge score at budget 1
        self.b1_kcomp = int(getattr(cfg, 'b1_kcomp', 50))  # competitor-set size for RA-IF
        # b1_signal: victim-free single-edge selector at budget 1: raif (competitor-margin) |
        # drift (exact surrogate rank-drop) | effres (effective-resistance bottleneck) | off
        # NB: YAML/Hydra parse `off` as boolean False -> normalize all off-like values to 'off'
        _b1s = getattr(cfg, 'b1_signal', None)
        if _b1s is True:
            _b1s = 'raif'
        if _b1s in (None, False, '', 'off', 'False', 'false', 'None'):
            _b1s = 'raif' if getattr(cfg, 'b1_margin', False) else 'off'
        self.b1_signal = str(_b1s)
        self._adapt_cache = {}
        deg = torch.bincount(self._edge_index[0], minlength=n).float()
        self._deg = deg
        self._deg_med = float(deg[deg > 0].median().item()) if int((deg > 0).sum()) else 1.0
        if self.bias_signal == 'thp' and self.degree_bias != 0.0:
            ei_cpu = self._edge_index.cpu()
            self._adj = [set() for _ in range(n)]
            s_, d_ = ei_cpu[0].tolist(), ei_cpu[1].tolist()
            for a, b in zip(s_, d_):
                self._adj[a].add(b)
            self._xn = torch.nn.functional.normalize(self._x.float(), p=2, dim=1, eps=1e-8)
            self._thp_cache = {}

        # Per-surrogate extracted weights + precomputed xW1 (for ensembled gradients).
        self._surr_w = []
        for model, _ in self._surrogate_runs:
            W1 = model.W1.weight.detach(); b1 = model.W1.bias.detach()
            W2 = model.W2.weight.detach(); b2 = model.W2.bias.detach()
            self._surr_w.append({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
                                 'xW1': (self._x @ W1.t()).detach()})
        self._activate_surrogate_run(0)
        self._order_cache = self._order_cache_by_run[0]

        logger.info(
            f"CLEAVE initialized (self-supervised surrogate x{self.n_surrogates}, victim-independent): "
            f"loss={self.embedding_loss_type}, T={self.num_gradient_modifications}, "
            f"top_k={self.top_k}, E={self._E}, N={n}"
        )

    def _b1_margin_scores(self, t, kcomp=None):
        """RA-IF per-edge scores for budget 1: returns (nb_list, scores_tensor) where scores are the
        rank-aligned competitor-margin = (target sim-to-query drop) + (mean RISE of K bracketing
        competitors), averaged over the ensemble. Plus target_drop and competitor_rise components."""
        kcomp = int(getattr(self, 'b1_kcomp', 50)) if kcomp is None else kcomp
        import torch.nn.functional as F
        ei = self._edge_index
        nbrs = torch.unique(torch.cat([ei[1][ei[0] == t], ei[0][ei[1] == t]]))
        if nbrs.numel() == 0:
            return [], None, None, None
        nb_list = nbrs.tolist()
        scores = torch.zeros(len(nb_list), device=self.device)
        tdrop = torch.zeros(len(nb_list), device=self.device)
        crise = torch.zeros(len(nb_list), device=self.device)
        for run in range(self.n_surrogates):
            self._use_surrogate(run)
            q = self._centroid_query(t)
            zc = self._gcn_forward_sparse(torch.ones(self._E, device=self.device)).detach()
            simc = F.cosine_similarity(q, zc, dim=1)
            st = simc[t]
            order = torch.argsort((simc - st).abs())          # nodes whose sim brackets the target
            comp = order[order != t][:kcomp]
            for i, j in enumerate(nb_list):
                ev = torch.ones(self._E, device=self.device)
                ev[((ei[0] == t) & (ei[1] == j)) | ((ei[0] == j) & (ei[1] == t))] = 0
                simd = F.cosine_similarity(q, self._gcn_forward_sparse(ev).detach(), dim=1)
                td = (simc[t] - simd[t]).item(); cr = (simd[comp] - simc[comp]).mean().item()
                tdrop[i] += td; crise[i] += cr; scores[i] += td + cr
        self._use_surrogate(0)
        return nb_list, scores / self.n_surrogates, tdrop / self.n_surrogates, crise / self.n_surrogates

    def _b1_margin_edge(self, t, kcomp=None):
        """RA-IF single best edge (budget=1): argmax of the competitor-margin score. Victim-free."""
        nb_list, scores, _, _ = self._b1_margin_scores(t, kcomp)
        if not nb_list:
            return []
        return [(t, nb_list[int(torch.argmax(scores).item())])]

    def get_target_edges(self, target_node):
        """Incident edges of the target. Under graph_fraction<1, restrict to the SPARSIFIED edge set
        (so Module-2's feature-similarity fill cannot reach edges CLEAVE was not given). Full graph otherwise."""
        if getattr(self, '_sparse_ei', None) is None:
            return super().get_target_edges(target_node)
        t = int(target_node); ei = self._edge_index
        mask = (ei[0] == t) | (ei[1] == t)
        cols = ei[:, mask].t().tolist()
        out, seen = [], set()
        for a, b in cols:
            nb = b if a == t else a
            if nb == t or nb in seen:
                continue
            seen.add(nb); out.append((t, nb))
        return out

    def _b1_incident(self, t):
        ei = self._edge_index
        nbrs = torch.unique(torch.cat([ei[1][ei[0] == t], ei[0][ei[1] == t]]))
        return nbrs.tolist()

    def _b1_drift_edge(self, t):
        """EXACT single-edge surrogate rank-drop (budget=1): pick the incident edge whose removal most
        increases t's rank vs the frozen centroid query in the SURROGATE's space -- the literal surrogate
        b1 optimum (RA-IF only approximates this with a margin). Ensemble-averaged. Victim-free."""
        nb_list = self._b1_incident(t)
        if not nb_list:
            return []
        scores = torch.zeros(len(nb_list), device=self.device)
        for run in range(self.n_surrogates):
            self._use_surrogate(run)
            for i, nb in enumerate(nb_list):
                scores[i] += self._surrogate_rank_drift(t, {nb})
        self._use_surrogate(0)
        return [(t, nb_list[int(torch.argmax(scores).item())])]

    def _b1_effres_edge(self, t):
        """Effective-resistance single-edge (budget=1): pick the incident edge with the largest effective
        resistance on t's 2-hop ego graph -- a victim-free STRUCTURAL bottleneck measure (load-bearing
        edge), orthogonal to the embedding-gradient signal. R_eff(t,nb)=L+[t,t]+L+[nb,nb]-2 L+[t,nb]."""
        nb_list = self._b1_incident(t)
        if not nb_list:
            return []
        ei = self._ei_cpu if hasattr(self, '_ei_cpu') else self._edge_index.cpu()
        # 2-hop ego node set
        import numpy as np
        s = ei[0].numpy(); d = ei[1].numpy()
        from collections import defaultdict
        if not hasattr(self, '_adjlist'):
            adj = defaultdict(set)
            for a, b in zip(s.tolist(), d.tolist()):
                adj[a].add(b)
            self._adjlist = adj
        adj = self._adjlist
        ego = {t} | set(adj[t])
        for u in list(adj[t]):
            ego |= adj[u]
        ego = sorted(ego)
        idx = {n: i for i, n in enumerate(ego)}
        m = len(ego)
        L = np.zeros((m, m))
        for u in ego:
            for v in adj[u]:
                if v in idx:
                    L[idx[u], idx[v]] -= 1.0; L[idx[u], idx[u]] += 1.0
        Lp = np.linalg.pinv(L)
        it = idx[t]
        scores = []
        for nb in nb_list:
            j = idx.get(nb)
            scores.append(Lp[it, it] + Lp[j, j] - 2 * Lp[it, j] if j is not None else -1e9)
        return [(t, nb_list[int(np.argmax(scores))])]

    def _b1_combo_edge(self, t):
        """Combine the EXACT surrogate rank-drop (drift; best where rank-jumps are large) with the
        smooth competitor-MARGIN (RA-IF; best as a tie-breaker where rank-drops are quantized/tied):
        score = z(rank_drop) + b1_lam * z(margin), argmax. Victim-free, ensemble-averaged."""
        nb_list, margin, _, _ = self._b1_margin_scores(t)
        if not nb_list:
            return []
        drift = torch.zeros(len(nb_list), device=self.device)
        for run in range(self.n_surrogates):
            self._use_surrogate(run)
            for i, nb in enumerate(nb_list):
                drift[i] += self._surrogate_rank_drift(t, {nb})
        self._use_surrogate(0)
        def z(a):
            return (a - a.mean()) / (a.std() + 1e-9)
        lam = float(getattr(self, 'b1_lam', 1.0))
        comb = z(drift) + lam * z(margin)
        return [(t, nb_list[int(torch.argmax(comb).item())])]

    def _b1_select(self, t):
        sig = getattr(self, 'b1_signal', 'raif')
        if sig == 'drift':
            return self._b1_drift_edge(t)
        if sig == 'effres':
            return self._b1_effres_edge(t)
        if sig == 'combo':
            return self._b1_combo_edge(t)
        return self._b1_margin_edge(t)          # 'raif' default

    def _surrogate_rank_drift(self, t, nbset):
        """Victim-FREE drift proxy: how far target t falls in the SURROGATE's own retrieval
        (vs its frozen neighbourhood-centroid query) when the incident edges to nbset are removed.
        Used to choose, per target, between the degree-biased and plain removal sets."""
        import torch.nn.functional as F
        if not nbset:
            return -1e9
        q = self._centroid_query(t)
        run = self._active_surrogate_run
        zc = self._clean_emb_full_cache.get(run)
        if zc is None:
            zc = self._gcn_forward_sparse(torch.ones(self._E, device=self.device)).detach()
            self._clean_emb_full_cache[run] = zc
        def rank(Z):
            sim = F.cosine_similarity(q, Z, dim=1)
            return int((sim > sim[t]).sum().item())
        r0 = rank(zc)
        ev = torch.ones(self._E, device=self.device)
        ei = self._edge_index
        nbt = torch.as_tensor(list(nbset), device=self.device)
        drop = ((ei[0] == t) & torch.isin(ei[1], nbt)) | (torch.isin(ei[0], nbt) & (ei[1] == t))
        ev[drop] = 0
        zd = self._gcn_forward_sparse(ev).detach()
        return rank(zd) - r0

    def attack(self, target_node, budget, add_edge=False):
        """Adaptive: choose per target between the degree-biased and plain removal sets by the
        victim-free surrogate-drift proxy. Falls back to the parent (plain/biased) when disabled."""
        if int(budget) == 1 and (getattr(self, 'b1_margin', False) or getattr(self, 'b1_signal', 'off') != 'off'):
            edges = self._b1_select(int(target_node))
            return [self.data.clone() if not edges else self._remove_edges(edges) for _ in range(self.trials)]
        if not self.degree_adaptive or self.degree_bias == 0.0:
            return super().attack(target_node, budget, add_edge)
        t = int(target_node)
        max_b = max(budget, getattr(self.system_config.general, 'budget', 5))
        if t not in self._adapt_cache:
            order_deg = self._compute_attack_order(t, max_b)
            save = self.degree_bias
            self.degree_bias = 0.0
            try:
                order_plain = self._compute_attack_order(t, max_b)
            finally:
                self.degree_bias = save
            self._adapt_cache[t] = (order_deg, order_plain)
        order_deg, order_plain = self._adapt_cache[t]
        Sd = {self.get_edge_neighbor(e, t) for e in order_deg[:budget]}
        Sp = {self.get_edge_neighbor(e, t) for e in order_plain[:budget]}
        chosen = order_deg if self._surrogate_rank_drift(t, Sd) > self._surrogate_rank_drift(t, Sp) else order_plain
        edges = chosen[:budget]
        out = []
        for _ in range(self.trials):
            out.append(self.data.clone() if not edges else self._remove_edges(edges))
        return out

    def _bias_factors(self, target_node, neighbors):
        """Per-neighbour multiplicative selection weight from the chosen training-free signal.
        degree: (deg_j / deg_median)**beta.  thp: rank of THP(t->j) among the target's incident
        edges, mapped to (0.5 .. 1.5)**beta (rank-based => robust to THP's possible negatives)."""
        beta = self.degree_bias
        if self.bias_signal != 'thp':
            return {int(j): (self._deg[int(j)].item() / self._deg_med) ** beta for j in neighbors}
        import math
        from collections import Counter
        t = int(target_node)
        nbrs_t = self._adj[t]
        reach = Counter()
        for k in nbrs_t:
            for v in self._adj[k]:
                if v != t and v not in nbrs_t:
                    reach[v] += 1
        xt = self._xn[t]
        vals = {}
        for j in neighbors:
            j = int(j)
            excl = sum(1 for v in self._adj[j] if v != t and v not in nbrs_t and reach.get(v, 0) == 1)
            cos = float(torch.dot(xt, self._xn[j]).item())
            vals[j] = cos * (1 + excl) / math.sqrt(max(int(self._deg[j].item()), 1))
        order = sorted(vals, key=lambda j: vals[j])               # ascending THP
        nn = len(order)
        fac = {}
        for r, j in enumerate(order):
            rr = r / (nn - 1) if nn > 1 else 1.0                  # 0=lowest THP, 1=highest
            fac[j] = (0.5 + rr) ** beta
        return fac

    def _average_gradient_module(self, target_node, max_perturbations):
        """AGPOA Module 1 with optional DEGREE-AWARE selection (degree_bias>0). Identical to the
        parent when degree_bias==0; otherwise, among edges with a positive demotion score, picks the
        one maximizing score * (deg(neighbor)/deg_median)**degree_bias — biasing toward hub edges the
        victim is most sensitive to (the parent's gradient under-weights them on dense graphs)."""
        if self.degree_bias == 0.0:
            return super()._average_gradient_module(target_node, max_perturbations)
        import torch as _t
        edge_values = _t.ones(self._E, device=self.device)
        avg_grad = _t.zeros(self._E, device=self.device)
        perturbations = []
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
        all_nb = set(nb_to_out) | set(nb_to_in)
        bias_fac = self._bias_factors(target_node, all_nb)        # per-neighbour signal weight
        for _ in range(self.num_gradient_modifications):
            grad = self._compute_adj_gradient(target_node, edge_values)
            grad_l1 = grad.abs().sum()
            if grad_l1 > 0:
                avg_grad = self.momentum_factor * avg_grad + grad / grad_l1
            best_w = float('-inf'); best_j = -1; best_edge = None
            for j in all_nb:
                if j in removed:
                    continue
                out_grad = avg_grad[nb_to_out[j]].item() if j in nb_to_out else 0.0
                in_grad = avg_grad[nb_to_in[j]].item() if j in nb_to_in else 0.0
                raw = -out_grad - in_grad
                if raw <= 0:                                   # only remove demoting edges (as parent)
                    continue
                w = raw * bias_fac.get(j, 1.0)
                if w > best_w:
                    best_w = w; best_j = j
                    best_edge = (j, target_node) if in_grad <= out_grad and j in nb_to_in else (target_node, j)
            if best_j == -1:
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

    def _use_surrogate(self, k: int):
        """Switch the active surrogate weights (for ensembled gradient steps) without
        touching the order cache; sets _active_surrogate_run so the centroid query is
        keyed/computed per surrogate."""
        w = self._surr_w[k]
        self._W1, self._b1, self._W2, self._b2 = w['W1'], w['b1'], w['W2'], w['b2']
        self._xW1 = w['xW1']
        self._active_surrogate_run = k

    def _compute_adj_gradient(self, target_node, edge_values_current):
        """Demotion gradient, averaged across the K self-supervised surrogates
        (each L1-normalized first -> equal-weight consensus)."""
        if self.n_surrogates <= 1:
            return super()._compute_adj_gradient(target_node, edge_values_current)
        total = None
        for k in range(self.n_surrogates):
            self._use_surrogate(k)
            ev = edge_values_current.detach().clone().requires_grad_(True)
            out = self._gcn_forward_sparse(ev)
            loss = self._target_embedding_loss(out, target_node)
            loss.backward()
            g = ev.grad.detach()
            g = g / (g.abs().sum() + 1e-12)
            total = g if total is None else total + g
        self._use_surrogate(0)
        return total / self.n_surrogates
