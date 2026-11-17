"""Self-contained analytic propagation helpers for the CLEAVE-add shortlist.
Â^k X propagation + the surrogate-free analytic demotion score used to form the
anchor shortlist. Extracted so the proposed method has no dependency on exploratory modules.
"""
import torch
import torch.nn.functional as F
from src.oracle.query import neighborhood_centroid_query


def _norm_adj_matmul(x, edge_index, n, dev):
    """Return (Â X) with Â = D~^-1/2 (A+I) D~^-1/2, plus the degree vector d~ (= deg + 1)."""
    si = torch.arange(n, device=dev)
    ei = torch.cat([edge_index, torch.stack([si, si])], dim=1)      # add self-loops
    deg = torch.zeros(n, device=dev).scatter_add_(0, ei[0], torch.ones(ei.size(1), device=dev))
    dinv = deg.pow(-0.5); dinv[torch.isinf(dinv)] = 0.0
    w = dinv[ei[0]] * dinv[ei[1]]
    A = torch.sparse_coo_tensor(ei, w, (n, n)).coalesce()
    return torch.sparse.mm(A, x), deg   # deg here already includes the +1 self-loop => d~


def _propagate_k(x, edge_index, n, dev, k):
    """Â^k X : k-hop symmetric-normalized propagation (SGC / low-pass filter). d~ from first hop."""
    P = x
    dtil = None
    for _ in range(k):
        P, dtil = _norm_adj_matmul(P, edge_index, n, dev)
    return P, dtil


class _AnalyticMixin:
    """Builds the SpecDiff analytic per-candidate demotion score in Â^k X space (surrogate-free)."""
    def _init_analytic(self, cfg, data):
        self._ak = int(getattr(cfg, "prop_k", 2))
        self._radius = self._sysc.dataset.query_radius
        self._P, self._dtil = _propagate_k(self.x, self._clean_ei, self._n, self.device, self._ak)

    def _analytic_score(self, target_node):
        t = int(target_node)
        q = neighborhood_centroid_query(self._clean_ei, t, self._radius, self._P, self._n)  # [1,d]
        Pt = self._P[t]; dt = float(self._dtil[t])
        dil = (dt / (dt + 1.0)) ** 0.5
        inj_w = 1.0 / ((dt + 1.0) * (self._dtil + 1.0)).sqrt()
        new_t = dil * Pt.unsqueeze(0) + inj_w.unsqueeze(1) * self._P
        return (-F.cosine_similarity(new_t, q, dim=1)).float()       # [N] higher = more demotion
