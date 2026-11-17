"""Set-selection (joint b-edge) add-edge demotion attackers (brainstorm Tier D).

Unlike top-b independent scoring, these pick the b-edge SET jointly, so they capture edge interaction
(diminishing returns / collinearity) at b>=2. All surrogate-free, scored by the exact multi-edge
mean-shift in propagated Â^k X space.

Adding a SET S of non-edges {(t,j): j in S} to t (|S|=b), the exact Kipf-Welling aggregate is
    z_t(S) = sqrt(d~_t/(d~_t+b)) * (Â^kX)_t  +  sum_{j in S} P_j / sqrt((d~_t+b)(d~_j+1))
Demotion objective  D(S) = -cos(z_t(S), q_t)  (higher = better). We minimize cos, i.e. maximize D.

  SubModDemote  -- greedy maximization of D(S) with lazy (CELF-style) evaluation. Provable (1-1/e) for
                   the monotone-submodular surrogate; captures interaction cheaply.
  FacLocDisperse-- farthest-point selection: pick anti-aligned anchors that are also mutually DISPERSED
                   (span independent anti-directions) rather than collinear near-duplicates.
  ShapEdge      -- Monte-Carlo Shapley value of each candidate over size<=b coalitions of the shortlist;
                   take the top-b Shapley scorers (credits complementarity/redundancy).
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.attackers import register_attacker
from src.attackers.contrib.analytic2_addedge import _propagate_k
from src.attackers.contrib.analytic_addedge import _norm_adj_matmul
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class _SetBase(Attacker):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        dev = data.edge_index.device
        self._n = int(getattr(data, "num_nodes", 0) or data.x.size(0))
        self._x = data.x.to(dev).float()
        self._ei = data.edge_index.to(dev)
        self._radius = system_config.dataset.query_radius
        self._k = int(getattr(cfg, "prop_k", 2))
        self._cand_M = int(getattr(cfg, "cand_topM", 256))   # shortlist size for set search
        self._P, self._dtil = _propagate_k(self._x, self._ei, self._n, dev, self._k)
        self._dev = dev
        logger.info(f"{self.name}: set-selection base ready (N={self._n}, shortlist={self._cand_M})")

    def _q(self, t):
        return neighborhood_centroid_query(self._ei, t, self._radius, self._P, self._n).squeeze(0)  # [d]

    def _shortlist(self, t, mask):
        """Top-M anti-aligned candidates (by single-edge injection) to keep set search tractable."""
        q = self._q(t)
        inj = 1.0/((float(self._dtil[t])+1.0)*(self._dtil+1.0)).sqrt()
        s = -(self._P @ (q/(q.norm()+1e-12))) * inj
        s[mask] = float("-inf")
        M = min(self._cand_M, int(torch.isfinite(s).sum()))
        return torch.topk(s, M).indices, q

    def _z_of_set(self, t, S, q):
        dt = float(self._dtil[t]); b = len(S)
        dil = (dt/(dt+b))**0.5
        z = dil*self._P[t]
        if b:
            Sj = torch.tensor(S, device=self._dev)
            w = 1.0/((dt+b)*(self._dtil[Sj]+1.0)).sqrt()
            z = z + (w.unsqueeze(1)*self._P[Sj]).sum(0)
        return -F.cosine_similarity(z.unsqueeze(0), q.unsqueeze(0), dim=1).item()

    def score_candidates(self, target_node):
        # fallback single-edge score (used if base calls it); real work is in _attack_add
        t = int(target_node); q = self._q(t)
        inj = 1.0/((float(self._dtil[t])+1.0)*(self._dtil+1.0)).sqrt()
        return (-(self._P @ (q/(q.norm()+1e-12)))*inj).float()

    def score_edges(self, target_node):
        raise NotImplementedError("add-edge only")

    def _select_set(self, t, budget, cand, q):
        raise NotImplementedError

    def _attack_add(self, target_node, budget):
        import random
        if isinstance(target_node, torch.Tensor): target_node = int(target_node.item())
        out = []
        mask = self._add_candidate_mask(target_node)
        cand, q = self._shortlist(target_node, mask)
        for i in range(self.trials):
            torch.manual_seed(self.seed+i); np.random.seed(self.seed+i); random.seed(self.seed+i)
            S = self._select_set(target_node, budget, cand, q)
            edges = [(int(target_node), int(j)) for j in S]
            out.append(self._add_edge(edges))
        return out


class SubModDemoteAttack(_SetBase):
    """Greedy submodular maximization of D(S) = -cos(z_t(S), q_t) with lazy CELF marginal gains."""
    def _select_set(self, t, budget, cand, q):
        cand = cand.tolist(); S = []; base = self._z_of_set(t, S, q)
        # lazy greedy
        gains = [(self._z_of_set(t, [j], q) - base, j) for j in cand]
        gains.sort(reverse=True)
        while len(S) < budget and gains:
            top_gain, j = gains[0]
            recomputed = self._z_of_set(t, S+[j], q) - self._z_of_set(t, S, q)
            if len(gains) > 1 and recomputed < gains[1][0]:
                gains[0] = (recomputed, j); gains.sort(reverse=True); continue
            S.append(j); gains.pop(0)
        return S


class FacLocDisperseAttack(_SetBase):
    """Farthest-point selection over the anti-aligned shortlist: seed with the single most anti-aligned
    node, then iteratively add the candidate maximizing (anti-alignment) while FAR (in propagated space)
    from already-picked anchors — spans independent anti-directions instead of collinear duplicates."""
    def _select_set(self, t, budget, cand, q):
        Pc = self._P[cand]; qn = q/(q.norm()+1e-12)
        anti = -(Pc @ qn)                                     # [M]
        Pn = F.normalize(Pc, dim=1)
        S_idx = [int(torch.argmax(anti))]
        while len(S_idx) < min(budget, cand.numel()):
            picked = Pn[S_idx]                                # [s,d]
            sim_to_picked = (Pn @ picked.t()).max(1).values  # [M] closeness to nearest anchor
            score = anti - 1.0*sim_to_picked                 # far AND anti-aligned
            for si in S_idx: score[si] = float("-inf")
            S_idx.append(int(torch.argmax(score)))
        return [int(cand[i]) for i in S_idx]


class ShapEdgeAttack(_SetBase):
    """Monte-Carlo Shapley value of each shortlist candidate for the demotion objective; take top-b."""
    def __init__(self, cfg, system_config, data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data)
        self._mc = int(getattr(cfg, "shap_perms", 32))
        self._cand_M = min(self._cand_M, int(getattr(cfg, "shap_M", 48)))  # Shapley is O(perm*M*b)
    def _select_set(self, t, budget, cand, q):
        cl = cand.tolist(); M = len(cl); b = budget
        phi = {j: 0.0 for j in cl}
        for _ in range(self._mc):
            perm = cl[:]; np.random.shuffle(perm)
            S = []; prev = self._z_of_set(t, S, q)
            for j in perm[:max(b*4, b)]:                       # marginal contributions up to a coalition cap
                if len(S) >= b:
                    break
                cur = self._z_of_set(t, S+[j], q)
                phi[j] += (cur - prev); prev = cur; S.append(j)
        ranked = sorted(cl, key=lambda j: phi[j], reverse=True)
        return ranked[:b]


for nm, cls in [("submod_demote", SubModDemoteAttack), ("facloc_disperse", FacLocDisperseAttack),
                ("shapedge", ShapEdgeAttack)]:
    register_attacker(nm, cls)
ATTACKERS = {"submod_demote": SubModDemoteAttack, "facloc_disperse": FacLocDisperseAttack,
             "shapedge": ShapEdgeAttack}
