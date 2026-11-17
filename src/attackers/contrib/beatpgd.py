"""Tier-A methods aimed at BEATING PGD on demotion (brainstorm beatpgd_brainstorm.md).

All subclass PGDAddAttack, so the optimizer, projection, forward, surrogate, and rounding are
byte-identical to PGD -- only the loss (CWM/SoftRank) or the candidate signal (SpecFusion/Anchor)
changes, isolating each idea.

  RankPGD  -- replace 1-cos(z_t,q_t) with a listwise soft-rank / competitor-wall margin loss run
              through the SAME iterative PGD optimizer. Needs less surrogate fidelity: only the
              ORDERING near t's rank must transfer, not global cosine geometry. (Identical to cosine
              ONLY under one-shot; different when iterated -- the untested lever.)
  SpecFusionPGD -- each PGD step, blend on RANKS the surrogate mask-gradient with the surrogate-free
              SpecDiff analytic Â^kX score, carrying victim-correct signal PGD's surrogate lacks.
  AnchorPGD -- restrict PGD's candidate set to (SpecDiff top-K) U (surrogate-gradient top-K), so the
              surrogate's large-graph ranking errors only arbitrate among analytically-vetted edges.
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.published_addedge import PGDAddAttack
from src.attackers.contrib.analytic2_addedge import _propagate_k
from src.attackers import register_attacker
from src.oracle.query import neighborhood_centroid_query
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class RankPGDAttack(PGDAddAttack):
    """Competitor-wall margin / soft-rank demotion loss through the PGD optimizer."""
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.rank_mode = str(getattr(cfg, "rank_mode", "softrank"))   # softrank | hinge
        self.rank_tau = float(getattr(cfg, "rank_tau", 0.05))
        self.rank_topk = int(getattr(cfg, "rank_topk", 256))          # competitor "wall" size
        self.rank_margin = float(getattr(cfg, "rank_margin", 0.1))
        self._wall_cache = {}
        logger.info(f"RankPGD: mode={self.rank_mode} tau={self.rank_tau} topK={self.rank_topk} "
                    f"margin={self.rank_margin} (steps={self.pgd_steps}, lr={self.pgd_lr})")

    def _wall(self, target_node):
        w = self._wall_cache.get(target_node)
        if w is None:
            q = self._centroid_query(target_node)                     # [1,d]
            zc = self.clean_embeddings                                # [N,d] surrogate clean emb
            sims = F.cosine_similarity(zc, q, dim=1)                  # [N]
            k = min(self.rank_topk, zc.size(0))
            idx = torch.topk(sims, k).indices                        # top-K wall (high-rank competitors)
            w = (zc[idx].detach(), q.detach())
            self._wall_cache[target_node] = w
        return w

    def _demotion_loss(self, out, target_node):
        zc_wall, q = self._wall(target_node)
        z_t = out[target_node:target_node + 1]
        cos_t = F.cosine_similarity(z_t, q, dim=1)                    # [1]
        cos_w = F.cosine_similarity(zc_wall, q, dim=1)               # [K]
        if self.rank_mode == "hinge":
            # maximize total margin by which t sits BELOW each wall competitor
            return torch.clamp(self.rank_margin + (cos_w - cos_t), min=0).mean()
        # softrank: soft count of wall competitors that outrank t (higher = more demoted)
        return torch.sigmoid((cos_w - cos_t) / self.rank_tau).mean()


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


class SpecFusionPGDAttack(PGDAddAttack, _AnalyticMixin):
    """Blend surrogate PGD gradient with the analytic SpecDiff score on RANKS each step."""
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self._init_analytic(cfg, data)
        self.fuse_beta = float(getattr(cfg, "fuse_beta", 0.5))       # 0=pure surrogate, 1=pure analytic
        logger.info(f"SpecFusion-PGD: beta={self.fuse_beta}, prop_k={self._ak} (steps={self.pgd_steps})")

    def _rank(self, v):
        # dense rank in [0,1]: higher value -> higher rank
        return torch.argsort(torch.argsort(v)).float() / max(v.numel() - 1, 1)

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        n = self._n; budget = int(getattr(self._sysc.general, "budget", 5))
        j = torch.arange(n, device=self.device)
        tcol = torch.full((n,), target_node, device=self.device, dtype=torch.long)
        cand = torch.cat([torch.stack([tcol, j]), torch.stack([j, tcol])], dim=1)
        a_rank = self._rank(self._analytic_score(target_node))       # [N] analytic rank, fixed
        s = torch.full((n,), budget / max(n, 1), device=self.device)
        for _ in range(self.pgd_steps):
            s = s.detach().requires_grad_(True)
            out = self._forward_with_cands(cand, torch.cat([s, s]))
            loss = self._demotion_loss(out, target_node)
            (g,) = torch.autograd.grad(loss, s)
            # blend the mask update direction on ranks: surrogate gradient + analytic prior
            g_rank = self._rank(g)
            blended = (1 - self.fuse_beta) * g_rank + self.fuse_beta * a_rank
            s = self._project_box_budget(s + self.pgd_lr * (blended - blended.mean()), budget)
        # final selection score = rank blend of the converged mask and the analytic prior
        return ((1 - self.fuse_beta) * self._rank(s.detach()) + self.fuse_beta * a_rank).float()


class AnchorPGDAttack(PGDAddAttack, _AnalyticMixin):
    """Restrict the PGD candidate set to (analytic top-K) U (one-step surrogate-gradient top-K)."""
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self._init_analytic(cfg, data)
        self.anchor_K = int(getattr(cfg, "anchor_K", 128))
        logger.info(f"AnchorPGD: K={self.anchor_K} (analytic ∪ surrogate top-K), steps={self.pgd_steps}")

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        n = self._n; budget = int(getattr(self._sysc.general, "budget", 5))
        mask = self._add_candidate_mask(target_node)
        # analytic top-K
        a = self._analytic_score(target_node).clone(); a[mask] = float("-inf")
        aK = torch.topk(a, min(self.anchor_K, int(torch.isfinite(a).sum()))).indices
        # one-step surrogate-gradient top-K
        j = torch.arange(n, device=self.device)
        tcol = torch.full((n,), target_node, device=self.device, dtype=torch.long)
        cand_full = torch.cat([torch.stack([tcol, j]), torch.stack([j, tcol])], dim=1)
        s0 = torch.full((n,), budget / max(n, 1), device=self.device).requires_grad_(True)
        out = self._forward_with_cands(cand_full, torch.cat([s0, s0]))
        (g0,) = torch.autograd.grad(self._demotion_loss(out, target_node), s0)
        gm = g0.clone(); gm[mask] = float("-inf")
        gK = torch.topk(gm, min(self.anchor_K, int(torch.isfinite(gm).sum()))).indices
        # union shortlist
        M_idx = torch.unique(torch.cat([aK, gK]))
        M = M_idx.numel()
        cand_M = torch.cat([torch.stack([torch.full((M,), target_node, device=self.device), M_idx]),
                            torch.stack([M_idx, torch.full((M,), target_node, device=self.device)])], dim=1)
        s = torch.full((M,), budget / max(M, 1), device=self.device)
        for _ in range(self.pgd_steps):
            s = s.detach().requires_grad_(True)
            out = self._forward_with_cands(cand_M, torch.cat([s, s]))
            (g,) = torch.autograd.grad(self._demotion_loss(out, target_node), s)
            s = self._project_box_budget(s + self.pgd_lr * g, budget)
        full = torch.full((n,), float("-inf"), device=self.device)
        full[M_idx] = s.detach().float()
        return full


for nm, cls in [("rank_pgd", RankPGDAttack), ("specfusion_pgd", SpecFusionPGDAttack),
                ("anchor_pgd", AnchorPGDAttack)]:
    register_attacker(nm, cls)
ATTACKERS = {"rank_pgd": RankPGDAttack, "specfusion_pgd": SpecFusionPGDAttack, "anchor_pgd": AnchorPGDAttack}
