"""Tier-A improvements to GreedyAnchor, aimed at the surrogate-TRANSFER losses on large graphs.

v2 (bigger net + polish) was a no-op because it stayed inside the surrogate, and greedy already holds the
surrogate's favorites. These two instead change the SELECTION SIGNAL / OBJECTIVE:

  greedy_blend  -- each greedy marginal blends the SURROGATE set-loss with the surrogate-free ANALYTIC
                   set-score (Kipf-Welling multi-edge shift in A^k X space). The analytic signal transfers
                   BETTER than the surrogate on exactly the large graphs we lose (gate: PubMed 0.52>0.46,
                   DBLP 0.35>0.17). lambda is fixed OR "auto": low analytic<->surrogate agreement => trust
                   analytic more. Blend is on RANKS (scale-free).

  greedy_rank   -- greedy marginal uses the RANK / competitor-wall margin loss (RankPGD's _demotion_loss)
                   instead of 1-cos. Rank is what the metric rewards, and may need less surrogate fidelity.

Both keep GreedyAnchor's add-recompute-rescore loop; only the per-candidate score changes. Query-free.
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.greedy_anchor import GreedyAnchorAttack
from src.oracle.query import neighborhood_centroid_query
from src.attackers import register_attacker
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def _ranknorm(vals):
    """Map a list of floats to rank in [0,1] (higher value -> higher rank)."""
    a = np.asarray(vals, dtype=float)
    order = a.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(a))
    return r / max(len(a) - 1, 1)


class GreedyBlendAttack(GreedyAnchorAttack):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        lam = getattr(cfg, "blend_lambda", "auto")
        self.blend_lambda = lam  # float in [0,1] or "auto"
        logger.info(f"GreedyBlend: lambda={self.blend_lambda} (analytic+surrogate blended greedy marginal)")

    def _q_analytic(self, t):
        return neighborhood_centroid_query(self._clean_ei, t, self._radius, self._P, self._n).squeeze(0)

    def _analytic_set_score(self, t, chosen, q):
        """-cos(z_t(S), q) via the exact Kipf-Welling multi-edge shift in A^k X space."""
        dt = float(self._dtil[t]); b = len(chosen)
        dil = (dt / (dt + b)) ** 0.5 if b > 0 else 1.0
        z = dil * self._P[t]
        if b > 0:
            Sj = torch.tensor(chosen, device=self.device)
            w = 1.0 / ((dt + b) * (self._dtil[Sj] + 1.0)).sqrt()
            z = z + (w.unsqueeze(1) * self._P[Sj]).sum(0)
        return float(-F.cosine_similarity(z.unsqueeze(0), q.unsqueeze(0), dim=1))

    def _auto_lambda(self, target_node, shortlist, q):
        """Low surrogate<->analytic agreement over the shortlist -> lean analytic."""
        surr = self._analytic_score  # placeholder; we need surrogate single-edge scores
        # surrogate single-edge marginal (set-loss of {j}) vs analytic single-edge score
        s_surr = [self._set_loss(target_node, [j]) for j in shortlist]
        s_ana  = [self._analytic_set_score(target_node, [j], q) for j in shortlist]
        rs, ra = _ranknorm(s_surr), _ranknorm(s_ana)
        # Spearman = Pearson on ranks
        if np.std(rs) < 1e-9 or np.std(ra) < 1e-9:
            rho = 0.0
        else:
            rho = float(np.corrcoef(rs, ra)[0, 1])
        return float(np.clip(0.5 * (1.0 - rho), 0.0, 0.8)), (s_surr, s_ana)

    def _attack_add(self, target_node, budget):
        import random
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        out = []
        for tr in range(self.trials):
            torch.manual_seed(self.seed + tr); np.random.seed(self.seed + tr); random.seed(self.seed + tr)
            shortlist = self._shortlist(target_node, budget).tolist()
            q = self._q_analytic(target_node)
            if self.blend_lambda == "auto":
                lam, _ = self._auto_lambda(target_node, shortlist, q)
            else:
                lam = float(self.blend_lambda)
            chosen = []
            for _ in range(budget):
                cands = [j for j in shortlist if j not in chosen]
                if not cands: break
                Ls = [self._set_loss(target_node, chosen + [j]) for j in cands]         # surrogate
                La = [self._analytic_set_score(target_node, chosen + [j], q) for j in cands]  # analytic
                rs, ra = _ranknorm(Ls), _ranknorm(La)
                blended = (1.0 - lam) * rs + lam * ra
                chosen.append(cands[int(blended.argmax())])
            edges = [(int(target_node), int(j)) for j in chosen]
            out.append(self._add_edge(edges))
        return out


class GreedyRankAttack(GreedyAnchorAttack):
    """Greedy with the competitor-wall margin loss as the marginal (RankPGD's loss inside the greedy loop)."""
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.rank_mode = str(getattr(cfg, "rank_mode", "softrank"))
        self.rank_tau = float(getattr(cfg, "rank_tau", 0.05))
        self.rank_topk = int(getattr(cfg, "rank_topk", 256))
        self.rank_margin = float(getattr(cfg, "rank_margin", 0.1))
        self._wall_cache = {}
        logger.info(f"GreedyRank: mode={self.rank_mode} tau={self.rank_tau} topK={self.rank_topk}")

    def _wall(self, target_node):
        w = self._wall_cache.get(target_node)
        if w is None:
            q = self._centroid_query(target_node)
            zc = self.clean_embeddings
            k = min(self.rank_topk, zc.size(0))
            idx = torch.topk(F.cosine_similarity(zc, q, dim=1), k).indices
            w = (zc[idx].detach(), q.detach()); self._wall_cache[target_node] = w
        return w

    def _demotion_loss(self, out, target_node):
        zc_wall, q = self._wall(target_node)
        z_t = out[target_node:target_node + 1]
        cos_t = F.cosine_similarity(z_t, q, dim=1)
        cos_w = F.cosine_similarity(zc_wall, q, dim=1)
        if self.rank_mode == "hinge":
            return torch.clamp(self.rank_margin + (cos_w - cos_t), min=0).mean()
        return torch.sigmoid((cos_w - cos_t) / self.rank_tau).mean()


register_attacker("greedy_blend", GreedyBlendAttack)
register_attacker("greedy_rank", GreedyRankAttack)
ATTACKERS = {"greedy_blend": GreedyBlendAttack, "greedy_rank": GreedyRankAttack}
