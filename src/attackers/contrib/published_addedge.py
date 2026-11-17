"""Published add-edge baselines (second external wave).

Five attackers from the published-baselines assessment, adapted to the
framework's query-free, add-edge, surrogate-transfer rank-demotion setting via
the base ``score_candidates`` hook (or a PGD ``attack`` override). All operate on
candidate NON-edges {target, j}; the base ``_attack_add`` masks the target and
its current neighbors, takes the top-``budget`` on the GPU, and inserts them
symmetrically.

Attackers:
  dice_add    — DICE (Waniek et al., Nat Hum Behav 2018): connect the target to
                nodes OUTSIDE its (pseudo-label) community. Query-free/label-free.
  spac_add    — SPAC / Spectral Distance (Lin et al., KDD 2022): score each
                candidate non-edge by the change it induces in the graph Laplacian
                spectrum. Structure-only, query-free.
                margin structure score, extended to candidate non-edges (label
                from the surrogate pseudo-label, so label-free).
                relaxation of the candidate-edge mask + projected gradient descent
                on the surrogate demotion loss, rounded to the budget.
  pga_add     — PGA (Wang et al., TKDE 2024): efficient partial-graph gradient
                add-edge over a candidate shortlist (partial-access ablation).
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.attackers.surrogate_retriever import get_or_train_surrogate_retriever
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def _pseudo_labels(clean_emb, C, seed):
    """KMeans pseudo-labels on surrogate clean embeddings (label-free, as VIKING)."""
    zc = clean_emb.detach().cpu().numpy()
    C = max(2, min(C, zc.shape[0]))
    return KMeans(n_clusters=C, random_state=seed, n_init=10).fit_predict(zc), C


class _SurrogateBase(Attacker):
    """Shared init: trains/loads the retriever surrogate (transfer, black-box),
    caches clean all-node embeddings and the frozen centroid query. Mirrors the
    VIKING / Nettack-Retriever surrogate block so these baselines are consistent
    with the existing suite."""

    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        if retriever is None:
            raise ValueError(f"{type(self).__name__} requires a retriever")
        super().__init__(cfg, system_config, data)
        self.sort_reverse = True
        self._sysc = system_config
        surrogate, clean_embeddings = get_or_train_surrogate_retriever(
            data, retriever, system_config.dataset.name, system_config.retriever.name,
            graph_model=getattr(system_config.general, "graph_model", "gcn"),
            cache_dir=getattr(system_config.general, "cache_dir", "cache"),
            hidden_channels=getattr(cfg, "surrogate_hidden", 256),
            epochs=getattr(cfg, "surrogate_epochs", 200),
            lr=getattr(cfg, "surrogate_lr", 0.01),
            weight_decay=getattr(cfg, "surrogate_weight_decay", 5e-4),
            dropout=getattr(cfg, "surrogate_dropout", 0.0),
            loss_type=getattr(cfg, "surrogate_loss_type", "cosine"),
            patience=getattr(cfg, "surrogate_patience", 30),
            device=str(self.device), run_id=None, seed=self.seed,
        )
        self.retriever_surrogate_time = float(getattr(surrogate, "retriever_surrogate_time", 0.0))
        surrogate.eval()
        for p in surrogate.parameters():
            p.requires_grad_(False)
        self.surrogate = surrogate
        self.clean_embeddings = clean_embeddings.to(self.device)
        self._n = data.num_nodes
        self._clean_ei = data.edge_index.to(self.device)
        self.x = data.x.to(self.device).float()
        self._query_cache = {}

    def score_edges(self, target_node):
        """Add-edge-only baseline: removal scoring is not defined. Raises if run in
        remove mode (these attackers are meant for add_edge: true)."""
        raise NotImplementedError(
            f"{type(self).__name__} is an add-edge-only baseline; run with add_edge: true."
        )

    def _centroid_query(self, target_node):
        q = self._query_cache.get(target_node)
        if q is None:
            q = neighborhood_centroid_query(
                self._clean_ei, target_node,
                self._sysc.dataset.query_radius, self.clean_embeddings, self._n,
            ).detach()
            self._query_cache[target_node] = q
        return q


class DICEAddAttack(_SurrogateBase):
    """DICE add-edge: connect the target to nodes OUTSIDE its community.

    DICE ("delete internally, connect externally") is the canonical demoting-
    addition primitive. Here communities are surrogate KMeans pseudo-labels
    (label-free, transferable). The candidate score is +1 for every out-of-
    community node and -inf for same-community nodes, with a small feature-
    dissimilarity tiebreak so the most dissimilar external nodes are added first.
    Purely structural / query-free.
    """

    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        C = int(getattr(cfg, "num_pseudo_classes", 7))
        labels, C = _pseudo_labels(self.clean_embeddings, C, self.seed)
        self._labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        # feature-dissimilarity tiebreak (cosine to target features), normalized to (0,1)
        self._xn = F.normalize(self.x, p=2, dim=1)
        logger.info(f"DICE-Add initialised: pseudo-labels C={C}, N={self._n}")

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        same = (self._labels == self._labels[target_node])          # [N] bool
        # feature dissimilarity in (0,1): 1 - cos, mapped to [0,1]
        cos = (self._xn @ self._xn[target_node])                    # [N] in [-1,1]
        dissim = (1.0 - cos) * 0.5                                   # [N] in [0,1]
        score = dissim.clone()
        score[same] = float("-inf")                                 # only out-of-community candidates
        return score.float()


class SPACAddAttack(_SurrogateBase):
    """SPAC / Spectral Distance add-edge (Lin et al., KDD 2022).

    Scores each candidate non-edge {t, j} by the spectral perturbation it induces
    on the normalized graph Laplacian. Adding (t,j) is a rank-2 update of the
    adjacency; the first-order change in the Laplacian spectral energy along the
    top eigenvectors is a closed-form, structure-only signal. We use the standard
    first-order eigenvalue-perturbation proxy: the change in the Rayleigh energy
    Σ_k (φ_k[t] - φ_k[j])^2 over the leading K Laplacian eigenvectors φ_k, which is
    exactly the increase in graph "spectral distance" a new edge (t,j) causes.
    Larger perturbation => added first. Query-free.
    """

    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.n_eig = int(getattr(cfg, "n_eig", 64))
        self._phi, self._lam = self._compute_spectrum()
        logger.info(f"SPAC-Add initialised: K={self._phi.size(1)} Laplacian eigvecs, N={self._n}")

    def _compute_spectrum(self):
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        n = self._n
        ei = self._clean_ei.cpu().numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.float64)
        deg = np.asarray(A.sum(1)).ravel()
        dinv = np.zeros_like(deg); nz = deg > 0; dinv[nz] = deg[nz] ** -0.5
        Dinv = sp.diags(dinv)
        L = sp.eye(n) - Dinv @ A @ Dinv                    # normalized Laplacian
        k = min(self.n_eig, n - 2)
        lam, phi = eigsh(L, k=k, which="SM")               # smallest-magnitude eigenpairs
        return (torch.tensor(phi, dtype=torch.float32, device=self.device),
                torch.tensor(lam, dtype=torch.float32, device=self.device))

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        # spectral distance contribution of a new edge (t,j):
        # Σ_k (φ_k[t] - φ_k[j])^2  (weighted toward low-frequency modes GCN uses)
        phi_t = self._phi[target_node]                     # [K]
        diff = self._phi - phi_t.unsqueeze(0)              # [N, K]
        w = 1.0 / (self._lam + 1e-3)                       # emphasize low-frequency (smooth) modes
        return (diff.pow(2) * w.unsqueeze(0)).sum(dim=1).float()


class _GradSurrogateBase(_SurrogateBase):
    """Adds a differentiable augmented-adjacency GCN forward over candidate
    non-edges, reused by the PGA gradient attacker. Caches the linear-
    layer weights and a candidate-free sparse edge set."""

    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        s = self.surrogate
        self._W1 = s.W1.weight.detach(); self._b1 = s.W1.bias.detach()
        self._W2 = s.W2.weight.detach(); self._b2 = s.W2.bias.detach()
        ei = self._clean_ei
        no_self = ei[0] != ei[1]
        self._ei = ei[:, no_self]
        idx = torch.arange(self._n, device=self.device)
        self._loops = torch.stack([idx, idx])
        self._xW1 = (self.x @ self._W1.t()).detach()

    def _forward_with_cands(self, cand_edges, cand_vals):
        """Sparse normalized-GCN forward whose adjacency = existing edges (weight 1)
        + candidate directed edges (weight = cand_vals, differentiable) + self-loops.
        Returns all-node embeddings [N, d]."""
        n = self._n
        E = self._ei.shape[1]
        all_idx = torch.cat([self._ei, cand_edges, self._loops], dim=1)
        base = torch.ones(E, device=self.device)
        loops = torch.ones(n, device=self.device)
        vals = torch.cat([base, cand_vals, loops])
        adj = torch.sparse_coo_tensor(all_idx, vals, (n, n))
        deg = torch.sparse.mm(adj, torch.ones(n, 1, device=self.device)).squeeze(1)
        dis = deg.pow(-0.5); dis = torch.where(torch.isinf(dis), torch.zeros_like(dis), dis)
        row, col = all_idx
        norm = dis[row] * vals * dis[col]
        adj_norm = torch.sparse_coo_tensor(all_idx, norm, (n, n))
        h = torch.sparse.mm(adj_norm, self._xW1)
        if self._b1 is not None: h = h + self._b1
        h = F.relu(h)
        out = torch.sparse.mm(adj_norm, h @ self._W2.t())
        if self._b2 is not None: out = out + self._b2
        return out

    def _demotion_loss(self, out, target_node):
        z = out[target_node:target_node + 1]
        q = self._centroid_query(target_node)
        return 1.0 - F.cosine_similarity(z, q, dim=1).mean()


class PGAAddAttack(_GradSurrogateBase):
    """PGA (Wang et al., TKDE 2024) — efficient partial-graph gradient add-edge.

    Partial-graph access: the gradient signal is computed over a CANDIDATE
    SHORTLIST (the target's k-hop ball ∪ top feature-dissimilar nodes) rather
    than all N non-edges, modelling an attacker who only sees a neighborhood of
    the graph. A single backward of the demotion loss over the shortlisted
    candidate weights gives the per-node score; masked-out nodes score -inf.
    Maps directly onto the paper's partial-graph-access ablation.
    """

    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.khop = int(getattr(cfg, "khop", 2))
        self.shortlist = int(getattr(cfg, "shortlist", 512))
        self._xn = F.normalize(self.x, p=2, dim=1)

    def _candidate_pool(self, target_node):
        # k-hop ball
        frontier = {target_node}; seen = {target_node}
        ei = self._ei
        for _ in range(self.khop):
            nxt = set()
            for u in frontier:
                nbrs = torch.cat([ei[1, ei[0] == u], ei[0, ei[1] == u]]).tolist()
                nxt.update(nbrs)
            nxt -= seen; seen |= nxt; frontier = nxt
        pool = torch.tensor(sorted(seen), device=self.device, dtype=torch.long)
        # augment with top feature-dissimilar nodes (partial global view)
        cos = self._xn @ self._xn[target_node]
        far = torch.topk(-cos, min(self.shortlist, self._n)).indices
        return torch.unique(torch.cat([pool, far]))

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        pool = self._candidate_pool(target_node)                                   # [P]
        P = pool.shape[0]
        tcol = torch.full((P,), target_node, device=self.device, dtype=torch.long)
        cand = torch.cat([torch.stack([tcol, pool]), torch.stack([pool, tcol])], dim=1)
        vals = torch.zeros(2 * P, device=self.device, requires_grad=True)
        out = self._forward_with_cands(cand, vals)
        loss = self._demotion_loss(out, target_node)
        (g,) = torch.autograd.grad(loss, vals)
        pool_score = (g[:P] + g[P:]).detach()
        score = torch.full((self._n,), float("-inf"), device=self.device)
        score[pool] = pool_score.float()
        return score


ATTACKERS = {
    "dice_add": DICEAddAttack,
    "spac_add": SPACAddAttack,
    "pga_add": PGAAddAttack,
}
