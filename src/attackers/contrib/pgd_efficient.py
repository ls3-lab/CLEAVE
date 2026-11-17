"""PGD-Efficient: same surrogate + demotion loss as pgd_add, but cheaper to optimize.

Full pgd_add runs `pgd_steps` (default 100) projected-gradient steps, each a full-graph sparse GCN
forward+backward over ALL N candidate non-edges (~42 ms/step on CoraFull => ~4.4 s/target). Yet the
target keeps only b<=5 edges, so almost all N candidates are wasted compute every step.

Three levers (each toggleable so we can isolate the source of any speedup):
  1. SHORTLIST  -- one full-N gradient step ranks candidates by |dL/ds_j|; keep the top-M (>> budget).
                   The shortlist is in the EXACT surrogate-gradient space PGD optimizes, so it provably
                   contains the edges full PGD would climb toward. Remaining steps optimize only M<<N
                   candidates (smaller sparse forward + smaller gradient vector).
  2. WARM-START -- initialize the mask from that first gradient (budget mass on the top-budget picks)
                   instead of the uniform budget/N, so PGD starts near the optimum.
  3. EARLY-STOP -- halt when the demotion loss plateaus (|ΔL| < tol for `patience` steps).

The forward, loss, projection, and rounding are byte-identical to pgd_add — only the candidate SET and
the stopping rule change — so any quality change is attributable purely to the efficiency levers.
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.published_addedge import PGDAddAttack
from src.attackers import register_attacker
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class PGDEfficientAttack(PGDAddAttack):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.cand_M = int(getattr(cfg, "cand_topM", 1024))       # shortlist size (0 => disable, full N)
        self.warm_start = bool(getattr(cfg, "warm_start", False))
        self.early_tol = float(getattr(cfg, "pgd_early_tol", 1e-4))
        self.patience = int(getattr(cfg, "pgd_patience", 8))
        logger.info(f"PGD-Efficient: shortlist M={self.cand_M}, warm_start={self.warm_start}, "
                    f"early_tol={self.early_tol}, patience={self.patience}, max_steps={self.pgd_steps}")

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        n = self._n; dev = self.device
        budget = int(getattr(self._sysc.general, "budget", 5))
        j = torch.arange(n, device=dev)
        tcol = torch.full((n,), target_node, device=dev, dtype=torch.long)

        # ---- lever 1+2: one full-N gradient step -> shortlist + warm-start direction ----
        cand_full = torch.cat([torch.stack([tcol, j]), torch.stack([j, tcol])], dim=1)  # [2,2N]
        s0 = torch.full((n,), budget / max(n, 1), device=dev).requires_grad_(True)
        out = self._forward_with_cands(cand_full, torch.cat([s0, s0]))
        loss0 = self._demotion_loss(out, target_node)
        (g0,) = torch.autograd.grad(loss0, s0)                                          # [N] gradient
        mask = self._add_candidate_mask(target_node)

        if self.cand_M and self.cand_M < n:
            gm = g0.clone(); gm[mask] = float("-inf")
            M = min(self.cand_M, int(torch.isfinite(gm).sum()))
            M_idx = torch.topk(gm, M).indices                                           # [M]
        else:
            M_idx = (~mask).nonzero().flatten()                                          # all valid nodes
            M = M_idx.numel()

        cand_M = torch.cat([torch.stack([torch.full((M,), target_node, device=dev), M_idx]),
                            torch.stack([M_idx, torch.full((M,), target_node, device=dev)])], dim=1)
        # warm-start: concentrate budget on the top-budget gradient picks within the shortlist
        if self.warm_start:
            gsl = g0[M_idx]
            s = torch.full((M,), 1e-3, device=dev)
            topb = torch.topk(gsl, min(budget, M)).indices
            s[topb] = 1.0
            s = self._project_box_budget(s, budget)
        else:
            s = torch.full((M,), budget / max(M, 1), device=dev)

        # ---- PGD on the shortlist with early stopping ----
        prev = None; bad = 0; used = 0
        for step in range(self.pgd_steps):
            s = s.detach().requires_grad_(True)
            out = self._forward_with_cands(cand_M, torch.cat([s, s]))
            loss = self._demotion_loss(out, target_node)
            (g,) = torch.autograd.grad(loss, s)
            s = self._project_box_budget(s + self.pgd_lr * g, budget)
            used += 1
            L = float(loss.detach())
            if prev is not None and abs(L - prev) < self.early_tol:
                bad += 1
            else:
                bad = 0
            prev = L
            if bad >= self.patience:
                break
        self._last_steps_used = used

        full_score = torch.full((n,), float("-inf"), device=dev)
        full_score[M_idx] = s.detach().float()
        return full_score


register_attacker("pgd_eff", PGDEfficientAttack)
ATTACKERS = {"pgd_eff": PGDEfficientAttack}
