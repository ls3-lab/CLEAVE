"""Surrogate-free analytic add-edge demotion attackers (brainstorm Tier A / geometric family).

The victims are low-pass contrastive GCNs, so a target's retrieval position is (to first order) an
ANALYTIC function of its neighborhood aggregate. These attackers replace the trained gradient surrogate
with the GCN's OWN aggregation arithmetic -> no surrogate, no victim query, no transfer ceiling.

Shared proxy: P = (D~^-1/2 A~ D~^-1/2) X   (1-hop symmetric-normalized aggregate; A~ = A + I).
The frozen query proxy q_t = neighborhood-centroid of P over t's k-hop neighbors (same construction as
the evaluator's, but in the parameter-free proxy space).

Exact single-edge update (Kipf-Welling normalization), adding non-edge (t, j):
    (ÂX)_t' = sqrt(d~_t/(d~_t+1)) * (ÂX)_t   +   x_j / sqrt((d~_t+1)(d~_j+1))
             \_______ dilution (j-independent) ______/   \___ injection (rewards LOW-degree j) ___/
Demotion score = -cos((ÂX)_t', q_t): higher = pushes the target's aggregate further from its query.
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.attackers import register_attacker
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def _norm_adj_matmul(x, edge_index, n, dev):
    """Return (Â X) with Â = D~^-1/2 (A+I) D~^-1/2, plus the degree vector d~ (= deg + 1)."""
    si = torch.arange(n, device=dev)
    ei = torch.cat([edge_index, torch.stack([si, si])], dim=1)      # add self-loops
    deg = torch.zeros(n, device=dev).scatter_add_(0, ei[0], torch.ones(ei.size(1), device=dev))
    dinv = deg.pow(-0.5); dinv[torch.isinf(dinv)] = 0.0
    w = dinv[ei[0]] * dinv[ei[1]]
    A = torch.sparse_coo_tensor(ei, w, (n, n)).coalesce()
    return torch.sparse.mm(A, x), deg   # deg here already includes the +1 self-loop => d~


class _AnalyticBase(Attacker):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        dev = data.edge_index.device
        self._n = int(getattr(data, "num_nodes", 0) or data.x.size(0))
        self._x = data.x.to(dev).float()
        self._ei = data.edge_index.to(dev)
        self._radius = system_config.dataset.query_radius
        # 1-hop symmetric-normalized aggregate proxy embedding, and d~ (deg incl. self-loop)
        self._P, self._dtil = _norm_adj_matmul(self._x, self._ei, self._n, dev)
        self._dev = dev
        logger.info(f"{self.name}: analytic surrogate-free proxy P=ÂX built (N={self._n}, "
                    f"F={self._x.size(1)}, radius={self._radius})")

    def _q_proxy(self, target_node):
        return neighborhood_centroid_query(self._ei, target_node, self._radius, self._P, self._n)  # [1,F]

    def score_edges(self, target_node):
        raise NotImplementedError("analytic attackers are add-edge only")


class AMSPAttack(_AnalyticBase):
    """AMSP: exact 1-hop GCN-aggregation shift; pick edges maximizing the drop in proxy-cos(z_t, q_t)."""
    def score_candidates(self, target_node):
        t = int(target_node)
        q = self._q_proxy(t)                                   # [1,F]
        Pt = self._P[t]                                        # [F]  = (ÂX)_t
        dt = float(self._dtil[t])                              # d~_t
        dil = (dt / (dt + 1.0)) ** 0.5                         # dilution factor (j-independent)
        inj_w = 1.0 / ((dt + 1.0) * (self._dtil + 1.0)).sqrt() # [N] injection weight per candidate j
        # predicted new aggregate for every candidate j:  new_t[j] = dil*Pt + inj_w[j]*x_j
        new_t = dil * Pt.unsqueeze(0) + inj_w.unsqueeze(1) * self._x   # [N,F]
        score = -F.cosine_similarity(new_t, q, dim=1)          # higher = more demotion
        return score.float()


class AntiCentroidAttack(_AnalyticBase):
    """Degree-weighted anti-alignment: score j by (how anti-q_t its injected signal is) x injection weight.
    Surrogate-free one-shot; the AMSP ablation that drops the (constant) dilution term."""
    def score_candidates(self, target_node):
        t = int(target_node)
        q = self._q_proxy(t)                                   # [1,F]
        inj_w = 1.0 / (self._dtil + 1.0).sqrt()                # [N]  (drop the constant (d~_t+1) factor)
        anti = -F.cosine_similarity(self._x, q, dim=1)         # [N]  feature anti-alignment to query
        return (inj_w * anti).float()


register_attacker("amsp", AMSPAttack)
register_attacker("anticentroid", AntiCentroidAttack)
ATTACKERS = {"amsp": AMSPAttack, "anticentroid": AntiCentroidAttack}
