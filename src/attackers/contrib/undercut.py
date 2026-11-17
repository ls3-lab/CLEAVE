"""Undercut: query-free, black-box edge-addition rank-demotion attack (proposed method).

Trains a contrastive surrogate on the public graph, forms an anchor shortlist of candidate
endpoints (closed-form low-pass analytic score UNION one-step surrogate gradient), and selects
the b-edge set by a greedy add-recompute-rescore search. Each candidate marginal is evaluated by
an analytic degree-weighted update of the surrogate embeddings (no per-candidate forward pass):
    z_t(S) ~= sqrt(d_t/(d_t+|S|)) * Z_t + sum_{j in S} 1/sqrt((d_t+|S|)(d_j+1)) * Z_j
    loss   = 1 - cos(z_t(S), q_t)     (q_t = surrogate-space neighborhood centroid)
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.published_addedge import _GradSurrogateBase
from src.attackers.contrib._analytic_utils import _AnalyticMixin
from src.attackers import register_attacker
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class UndercutAttack(_GradSurrogateBase, _AnalyticMixin):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self._init_analytic(cfg, data)
        self.anchor_K = int(getattr(cfg, "anchor_K", 128))
        self.shortlist_mode = str(getattr(cfg, "shortlist_mode", "union"))   # ablation: union|analytic|surrogate
        self.selection_mode = str(getattr(cfg, "selection_mode", "greedy"))  # ablation: greedy|topb
        n = self._n
        deg = torch.zeros(n, device=self.device)
        deg.scatter_add_(0, self._clean_ei[0], torch.ones(self._clean_ei.size(1), device=self.device))
        self._deg = deg                                   # [N] clean node degrees
        self._Z = self.clean_embeddings                   # [N,d] surrogate clean embeddings
        self._qcache = {}
        logger.info("Undercut: greedy add-recompute-rescore with analytic degree-update marginals")

    def _q(self, t):
        q = self._qcache.get(t)
        if q is None:
            q = self._centroid_query(t).detach()          # [1,d] surrogate-space centroid
            self._qcache[t] = q
        return q

    def _set_loss(self, target_node, chosen):
        t = int(target_node); m = len(chosen)
        dt = float(self._deg[t])
        zt = ((dt / (dt + m)) ** 0.5) * self._Z[t]        # shrink existing aggregate
        if chosen:
            cj = torch.tensor(chosen, device=self.device, dtype=torch.long)
            inj = 1.0 / ((dt + m) * (self._deg[cj] + 1.0)).sqrt()   # [m]
            zt = zt + (inj.unsqueeze(1) * self._Z[cj]).sum(0)       # add injected neighbors
        return float(1.0 - F.cosine_similarity(zt.unsqueeze(0), self._q(t), dim=1))

    def _shortlist(self, target_node, budget):
        n = self._n; mask = self._add_candidate_mask(target_node)
        a = self._analytic_score(target_node).clone(); a[mask] = float("-inf")
        aK = torch.topk(a, min(self.anchor_K, int(torch.isfinite(a).sum()))).indices
        j = torch.arange(n, device=self.device)
        tcol = torch.full((n,), target_node, device=self.device, dtype=torch.long)
        cand_full = torch.cat([torch.stack([tcol, j]), torch.stack([j, tcol])], dim=1)
        s0 = torch.full((n,), budget / max(n, 1), device=self.device).requires_grad_(True)
        out = self._forward_with_cands(cand_full, torch.cat([s0, s0]))
        (g0,) = torch.autograd.grad(self._demotion_loss(out, target_node), s0)
        gm = g0.clone(); gm[mask] = float("-inf")
        gK = torch.topk(gm, min(self.anchor_K, int(torch.isfinite(gm).sum()))).indices
        if self.shortlist_mode == "analytic":  return torch.unique(aK)
        if self.shortlist_mode == "surrogate": return torch.unique(gK)
        return torch.unique(torch.cat([aK, gK]))

    def score_candidates(self, target_node):
        # Not used for selection (we override _attack_add), but keep a valid fallback = analytic score.
        return self._analytic_score(int(target_node))

    def _attack_add(self, target_node, budget):
        import random
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        out = []
        for tr in range(self.trials):
            torch.manual_seed(self.seed + tr); np.random.seed(self.seed + tr); random.seed(self.seed + tr)
            shortlist = self._shortlist(target_node, budget).tolist()
            chosen = []
            if self.selection_mode == "topb":
                scored = sorted(((self._set_loss(target_node, [j]), j) for j in shortlist), reverse=True)
                chosen = [j for _, j in scored[:budget]]
                edges = [(int(target_node), int(j)) for j in chosen]
                out.append(self._add_edge(edges)); continue
            for _ in range(budget):
                best_j, best_loss = None, -1e9
                for j in shortlist:
                    if j in chosen:
                        continue
                    L = self._set_loss(target_node, chosen + [j])   # add-recompute: exact set loss
                    if L > best_loss:
                        best_loss, best_j = L, j
                if best_j is None:
                    break
                chosen.append(best_j)                                 # commit, then rescore next round
            edges = [(int(target_node), int(j)) for j in chosen]
            out.append(self._add_edge(edges))
        return out


register_attacker("undercut", UndercutAttack)
ATTACKERS = {"undercut": UndercutAttack}
