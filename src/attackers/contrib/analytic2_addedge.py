"""More surrogate-free add-edge demotion attackers (brainstorm Tiers A-C).

SpecDiffAttack   -- AMSP scored in the LOW-PASS PROPAGATED space Â^k X (InvariantSurrogate/SGC filter)
                    instead of raw features. Directly tests whether AMSP failed because raw bag-of-words
                    geometry does not transfer, while the propagated low-frequency signal (shared by all
                    low-pass GCN victims) does.
EffResistAttack  -- effective-resistance / commute-distance bridges (spectral, surrogate-free), GATED by
                    feature anti-alignment (distant != q_t-opposed).
AntiCutAttack    -- opposing-community anchors: cluster on the propagated proxy, bridge t into the
                    community whose centroid is most anti-aligned with q_t (sharpened, gated VIKING).
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.attackers import register_attacker
from src.attackers.contrib.analytic_addedge import _norm_adj_matmul
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def _propagate_k(x, edge_index, n, dev, k):
    """Â^k X : k-hop symmetric-normalized propagation (SGC / low-pass filter). d~ from first hop."""
    P = x
    dtil = None
    for _ in range(k):
        P, dtil = _norm_adj_matmul(P, edge_index, n, dev)
    return P, dtil


class _PropBase(Attacker):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        dev = data.edge_index.device
        self._n = int(getattr(data, "num_nodes", 0) or data.x.size(0))
        self._x = data.x.to(dev).float()
        self._ei = data.edge_index.to(dev)
        self._radius = system_config.dataset.query_radius
        self._k = int(getattr(cfg, "prop_k", 2))          # propagation depth ~ victim receptive field
        self._P, self._dtil = _propagate_k(self._x, self._ei, self._n, dev, self._k)
        self._dev = dev
        logger.info(f"{self.name}: propagated proxy Â^{self._k} X built (N={self._n}, radius={self._radius})")

    def _q(self, t):
        return neighborhood_centroid_query(self._ei, t, self._radius, self._P, self._n)  # [1,d]

    def score_edges(self, target_node):
        raise NotImplementedError("add-edge only")


class SpecDiffAttack(_PropBase):
    """AMSP's exact-shift logic, but in the Â^k X propagated (low-pass) embedding space.
    Adding (t,j): new_t = dil*P_t + inj_w[j]*P_j  (P = propagated features act like the injected signal).
    Scored by -cos(new_t, q_t) in the shared low-frequency space all victims preserve."""
    def score_candidates(self, target_node):
        t = int(target_node)
        q = self._q(t)                                          # [1,d]
        Pt = self._P[t]                                         # [d]
        dt = float(self._dtil[t])
        dil = (dt / (dt + 1.0)) ** 0.5
        inj_w = 1.0 / ((dt + 1.0) * (self._dtil + 1.0)).sqrt()  # [N]
        new_t = dil * Pt.unsqueeze(0) + inj_w.unsqueeze(1) * self._P   # inject j's PROPAGATED signal
        return (-F.cosine_similarity(new_t, q, dim=1)).float()


class EffResistAttack(_PropBase):
    """Effective-resistance bridges gated by feature anti-alignment. R(t,j) estimated from the bottom-m
    Laplacian eigenpairs: R(t,j) = sum_l (1/lam_l)(u_l[t]-u_l[j])^2 . High R = diffusion-distant j (large
    single-edge rerouting); gate by -cos(P_j, q_t) so 'distant' also means 'query-opposed'."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        self._m = int(getattr(cfg, "eig_m", 64))
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        ei = self._ei.cpu().numpy(); n = self._n
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(float)
        deg = np.asarray(A.sum(1)).ravel()
        d_inv_sqrt = np.zeros(n); nz = deg > 0; d_inv_sqrt[nz] = deg[nz] ** -0.5
        Dn = sp.diags(d_inv_sqrt)
        Lnorm = sp.identity(n) - Dn @ A @ Dn           # normalized Laplacian
        m = min(self._m, n - 2)
        vals, vecs = eigsh(Lnorm.astype(float), k=m, which="SM")  # bottom-m
        keep = vals > 1e-6
        self._lam = torch.tensor(vals[keep], device=self._dev, dtype=torch.float32)   # [m']
        self._U = torch.tensor(vecs[:, keep], device=self._dev, dtype=torch.float32)  # [N, m']
        logger.info(f"{self.name}: bottom-{m} Laplacian eigenpairs ({int(keep.sum())} kept > 1e-6)")

    def score_candidates(self, target_node):
        t = int(target_node)
        q = self._q(t)
        du = self._U - self._U[t:t+1]                          # [N, m']  (u_l[j]-u_l[t])
        R = (du * du / self._lam.unsqueeze(0)).sum(1)          # [N]  effective resistance R(t,j)
        R = R / (R.max() + 1e-12)
        anti = (-F.cosine_similarity(self._P, q, dim=1)).clamp_min(0)  # [N] query anti-alignment gate
        return (R * anti).float()


class AntiCutAttack(_PropBase):
    """Opposing-community anchors. KMeans on the propagated proxy; per candidate j, score by how
    anti-aligned j's CLUSTER centroid is to q_t, times j's own anti-alignment and injection weight.
    Bridges t into the community whose mean signal is most opposed to the frozen query."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        self._nc = int(getattr(cfg, "n_clusters", 32))
        from sklearn.cluster import KMeans
        P = self._P.cpu().numpy()
        km = KMeans(n_clusters=min(self._nc, self._n), n_init=4, random_state=self.seed).fit(P)
        self._lab = torch.tensor(km.labels_, device=self._dev, dtype=torch.long)   # [N]
        self._cent = torch.tensor(km.cluster_centers_, device=self._dev, dtype=torch.float32)  # [C,d]
        logger.info(f"{self.name}: KMeans {self._cent.size(0)} clusters on Â^{self._k}X")

    def score_candidates(self, target_node):
        t = int(target_node)
        q = self._q(t)
        clu_anti = -F.cosine_similarity(self._cent, q, dim=1)  # [C] cluster-centroid anti-alignment
        node_anti = -F.cosine_similarity(self._P, q, dim=1)    # [N] node anti-alignment
        inj_w = 1.0 / (self._dtil + 1.0).sqrt()                # [N] low-degree injection bonus
        return (clu_anti[self._lab] * 0.5 + node_anti * 0.5).mul(inj_w).float()


for nm, cls in [("specdiff", SpecDiffAttack), ("effresist", EffResistAttack), ("anticut", AntiCutAttack)]:
    register_attacker(nm, cls)
ATTACKERS = {"specdiff": SpecDiffAttack, "effresist": EffResistAttack, "anticut": AntiCutAttack}
