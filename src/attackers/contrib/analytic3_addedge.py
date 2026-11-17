"""Remaining surrogate-free scoring-based add-edge demotion attackers (brainstorm Tiers A-D).

All reuse the propagated proxy (Â^k X) and/or bottom-m normalized-Laplacian eigenpairs. Each is a
one-shot score_candidates over all nodes; the base rounds to top-b. Query proxy q_t = neighborhood
centroid in the relevant proxy space (same construction as the evaluator).

  NettackInsertCF -- linear (SGC) closed-form insertion margin: maximize the drop in <q_t, z_t>.
  SpecTransfer    -- score in the bottom-m Laplacian eigenspace shared by all low-pass victims.
  IMDemote        -- influence-weighted: reward j whose k-hop receptive subtree (high reach) injects
                     an anti-aligned signal into t.
  FiedlerBridge   -- bridge across t's lowest-conductance cut (Fiedler vector sign), feature-gated.
  CurvatureBridge -- most negative Forman-Ricci edges (structural bridges), feature-gated.
  EigShift        -- first-order eigenvector-perturbation displacement of t away from q_t.
  InvariantSurrogate -- identity-SGC one-shot gradient (training-free low-pass encoder) demotion score.
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.attackers import register_attacker
from src.attackers.contrib.analytic_addedge import _norm_adj_matmul
from src.attackers.contrib.analytic2_addedge import _propagate_k
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class _SpecBase(Attacker):
    """Shared: propagated proxy Â^k X, degree d~, and lazy bottom-m normalized-Laplacian eigenpairs."""
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        dev = data.edge_index.device
        self._n = int(getattr(data, "num_nodes", 0) or data.x.size(0))
        self._x = data.x.to(dev).float()
        self._ei = data.edge_index.to(dev)
        self._radius = system_config.dataset.query_radius
        self._k = int(getattr(cfg, "prop_k", 2))
        self._P, self._dtil = _propagate_k(self._x, self._ei, self._n, dev, self._k)   # Â^k X
        self._P1, _ = _norm_adj_matmul(self._x, self._ei, self._n, dev)                 # Â X (1-hop)
        self._dev = dev
        self._eig = None
        logger.info(f"{self.name}: proxy Â^{self._k}X ready (N={self._n}, radius={self._radius})")

    def _q(self, t, P=None):
        P = self._P if P is None else P
        return neighborhood_centroid_query(self._ei, t, self._radius, P, self._n)

    def _eigpairs(self, m):
        if self._eig is None:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigsh
            ei = self._ei.cpu().numpy(); n = self._n
            A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
            A = ((A + A.T) > 0).astype(float)
            deg = np.asarray(A.sum(1)).ravel()
            dis = np.zeros(n); nz = deg > 0; dis[nz] = deg[nz] ** -0.5
            L = sp.identity(n) - sp.diags(dis) @ A @ sp.diags(dis)
            mm = min(m, n - 2)
            vals, vecs = eigsh(L.astype(float), k=mm, which="SM")
            self._eig = (torch.tensor(vals, device=self._dev, dtype=torch.float32),
                         torch.tensor(vecs, device=self._dev, dtype=torch.float32),
                         torch.tensor(deg, device=self._dev, dtype=torch.float32))
            logger.info(f"{self.name}: bottom-{mm} Laplacian eigenpairs computed")
        return self._eig

    def score_edges(self, target_node):
        raise NotImplementedError("add-edge only")


class NettackInsertCFAttack(_SpecBase):
    """Linear closed-form insertion margin (Nettack-style, W=I). Adding (t,j): z_t'=dil*z_t+inj*P_j.
    Maximize drop in the linear score <q,z_t>: j-dependent part = -inj_w[j]*<q,P_j>. Rank by that."""
    def score_candidates(self, target_node):
        t = int(target_node); q = self._q(t).squeeze(0)          # [d]
        dt = float(self._dtil[t]); inj_w = 1.0/((dt+1.0)*(self._dtil+1.0)).sqrt()
        return (-(self._P @ q) * inj_w).float()                  # higher = larger linear-margin drop


class SpecTransferAttack(_SpecBase):
    """Score AMSP's mean-shift but in the bottom-m eigenspace P_spec=U U^T X shared by low-pass victims."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        self._m = int(getattr(cfg, "eig_m", 64))
        _, U, _ = self._eigpairs(self._m)
        self._Ps = U @ (U.t() @ self._x)                         # [N,d] low-freq-filtered features
    def score_candidates(self, target_node):
        t = int(target_node); q = self._q(t, P=self._Ps)
        dt = float(self._dtil[t]); dil=(dt/(dt+1.0))**0.5
        inj_w = 1.0/((dt+1.0)*(self._dtil+1.0)).sqrt()
        new_t = dil*self._Ps[t].unsqueeze(0) + inj_w.unsqueeze(1)*self._Ps
        return (-F.cosine_similarity(new_t, q, dim=1)).float()


class IMDemoteAttack(_SpecBase):
    """Influence-weighted demotion: reward j whose k-hop receptive subtree has high REACH (propagated
    self-influence) and injects an anti-aligned signal. reach_j = (Â^k 1)_j ; score = reach*anti*inj."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        ones = torch.ones(self._n, 1, device=self._dev)
        r = ones
        for _ in range(self._k):
            r, _ = _norm_adj_matmul(r, self._ei, self._n, self._dev)
        self._reach = r.squeeze(1).clamp_min(0)                  # [N] propagated reach
    def score_candidates(self, target_node):
        t = int(target_node); q = self._q(t)
        anti = -F.cosine_similarity(self._P, q, dim=1)
        inj_w = 1.0/(self._dtil+1.0).sqrt()
        reach = self._reach/(self._reach.max()+1e-12)
        return (anti.clamp_min(0)*inj_w*(0.5+0.5*reach)).float()


class FiedlerBridgeAttack(_SpecBase):
    """Bridge t across its lowest-conductance cut. Use the smallest non-trivial eigenvector (Fiedler);
    reward j on the OPPOSITE side with large |f_j - f_t|, gated by feature anti-alignment."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        vals, U, _ = self._eigpairs(int(getattr(cfg, "eig_m", 64)))
        nz = (vals > 1e-6).nonzero().flatten()
        self._f = U[:, nz[0]] if nz.numel() else U[:, 0]         # Fiedler vector [N]
    def score_candidates(self, target_node):
        t = int(target_node); q = self._q(t)
        gap = (self._f - self._f[t]).abs()
        opp = (self._f.sign() != self._f[t].sign()).float()      # opposite side of the cut
        anti = (-F.cosine_similarity(self._P, q, dim=1)).clamp_min(0)
        return (gap*(0.5+0.5*opp)*anti).float()


class CurvatureBridgeAttack(_SpecBase):
    """Most negative Forman-Ricci edges = structural bridges between disjoint neighborhoods. Simplified
    unweighted Forman for a candidate (t,j): F = 4 - d_t - d_j + 3*|common neighbors|. Gate by anti-align."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        self._deg = self._dtil - 1.0                             # true degree (drop self-loop)
    def score_candidates(self, target_node):
        t = int(target_node); q = self._q(t)
        # common-neighbor count between t and every j via one sparse hop of the adjacency indicator
        nbr_t = torch.zeros(self._n, device=self._dev)
        row0 = self._ei[1][self._ei[0] == t]; nbr_t[row0] = 1.0
        # |common nbrs(t,j)| = (A[:,nbr_t]).sum -> approx via A @ nbr_t
        i = self._ei; vals = torch.ones(i.size(1), device=self._dev)
        A = torch.sparse_coo_tensor(i, vals, (self._n, self._n)).coalesce()
        common = torch.sparse.mm(A, nbr_t.unsqueeze(1)).squeeze(1)
        forman = 4.0 - self._deg[t] - self._deg + 3.0*common     # [N]
        neg = (-forman).clamp_min(0)                             # most negative curvature = largest bridge
        anti = (-F.cosine_similarity(self._P, q, dim=1)).clamp_min(0)
        return (neg*anti).float()


class EigShiftAttack(_SpecBase):
    """First-order eigenvector-perturbation displacement: adding (t,j) shifts t's spectral coordinates;
    reward edges that push t's low-freq coordinates toward those of ANTI-aligned regions (away from q_t)."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        vals, U, _ = self._eigpairs(int(getattr(cfg, "eig_m", 64)))
        keep = vals > 1e-6
        self._lam = vals[keep]; self._U = U[:, keep]            # [N,m']
    def score_candidates(self, target_node):
        t = int(target_node); q = self._q(t)
        # spectral coordinate distance between t and j weighted by 1/lam (low-freq dominates), gated
        du = self._U - self._U[t:t+1]
        disp = (du*du / self._lam.unsqueeze(0)).sum(1)          # [N]
        disp = disp/(disp.max()+1e-12)
        anti = (-F.cosine_similarity(self._P, q, dim=1)).clamp_min(0)
        return (disp*anti).float()


class InvariantSurrogateAttack(_SpecBase):
    """Training-free low-pass (SGC/identity) encoder as the CLEAVE-style one-shot gradient surrogate.
    Encoder z = Â^k X (no learned weights). Demotion gradient of 1-cos(z_t,q) wrt candidate edge value,
    computed in closed form via the mean-shift (identical family to SpecDiff but scored by gradient sign
    of the true cosine loss rather than the post-edge cosine)."""
    def score_candidates(self, target_node):
        t = int(target_node); q = self._q(t).squeeze(0)
        zt = self._P[t]
        zt_n = zt/ (zt.norm()+1e-12); q_n = q/(q.norm()+1e-12)
        dt = float(self._dtil[t]); inj_w = 1.0/((dt+1.0)*(self._dtil+1.0)).sqrt()
        # d/deps cos(zt+eps*inj*Pj, q) at eps=0  = inj*<Pj, q_n - cos0*zt_n>/|zt|
        cos0 = (zt_n*q_n).sum()
        dirv = q_n - cos0*zt_n
        grad = inj_w * (self._P @ dirv) / (zt.norm()+1e-12)     # d cos / d edge ; demotion wants NEGATIVE
        return (-grad).float()                                  # higher = most decreases cosine


for nm, cls in [("nettackinsert_cf", NettackInsertCFAttack), ("spectransfer", SpecTransferAttack),
                ("imdemote", IMDemoteAttack), ("fiedlerbridge", FiedlerBridgeAttack),
                ("curvaturebridge", CurvatureBridgeAttack), ("eigshift", EigShiftAttack),
                ("invariantsurrogate", InvariantSurrogateAttack)]:
    register_attacker(nm, cls)
ATTACKERS = {"nettackinsert_cf": NettackInsertCFAttack, "spectransfer": SpecTransferAttack,
             "imdemote": IMDemoteAttack, "fiedlerbridge": FiedlerBridgeAttack,
             "curvaturebridge": CurvatureBridgeAttack, "eigshift": EigShiftAttack,
             "invariantsurrogate": InvariantSurrogateAttack}
