"""CLEAVE-PGD++: CLEAVE contrastive surrogate + an improved PGD optimizer.

Diagnostics on the 4 biggest PGD-gap settings showed:
  - surrogate ENSEMBLE hurts (consensus dilutes the sharpest single-surrogate direction) -> dropped;
  - more PGD steps helps on all 4 (mean +10%, up to +22%);
  - the residual gap to PGD is a surrogate rank-transfer ceiling (Spearman 0.58-0.64 on large graphs).

CLEAVE-PGD++ keeps CLEAVE's query-free contrastive surrogate and PGD's joint edge optimization,
but improves the OPTIMIZER: (1) Adam-style momentum step instead of fixed-lr GD, (2) more steps,
(3) R random restarts keeping the mask with the best (highest) demotion loss. Loss and top-budget
rounding are unchanged. Everything stays query-free / victim-free (same access as CLEAVE-PGD).
Set pgd_restarts=1, pgd_optim='gd', pgd_steps=100 to recover plain CLEAVE-PGD exactly.
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.cleave_pgd import CLEAVEPGDAttack
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class CLEAVEPGDppAttack(CLEAVEPGDAttack):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        # Adaptive surrogate: the diagnostics traced the residual gap to a surrogate rank-transfer
        # ceiling on large graphs. Scale contrastive-surrogate training with graph size BEFORE
        # CLEAVE's init trains it (larger graphs -> more epochs + larger NT-Xent batch). Cache keys
        # on epochs/batch/hidden, so adaptive surrogates cache under distinct filenames (no clash).
        if bool(getattr(cfg, "adaptive_surrogate", True)):
            from omegaconf import open_dict
            n = int(getattr(data, "num_nodes", 0) or data.x.size(0))
            # epochs: 300 (<=5k nodes) -> up to 800 (>=30k); batch: 2048 -> min(n, 8192)
            ad_epochs = int(min(800, 300 + max(0, (n - 5000)) // 50))
            ad_batch = int(min(max(2048, n), 8192))
            try:
                with open_dict(cfg):
                    cfg.surrogate_epochs = ad_epochs
                    cfg.cl_batch = ad_batch
            except Exception:
                cfg.surrogate_epochs = ad_epochs; cfg.cl_batch = ad_batch
            logger.info(f"CLEAVE-PGD++ adaptive surrogate: epochs={ad_epochs}, cl_batch={ad_batch} (N={n})")
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.pgd_restarts = int(getattr(cfg, "pgd_restarts", 4))
        self.pgd_optim = str(getattr(cfg, "pgd_optim", "adam"))   # 'adam' | 'gd'
        self.adam_b1 = float(getattr(cfg, "adam_b1", 0.9))
        self.adam_b2 = float(getattr(cfg, "adam_b2", 0.999))
        logger.info(f"CLEAVE-PGD++: {self.pgd_steps} steps, {self.pgd_restarts} restarts, "
                    f"optim={self.pgd_optim}, lr={self.pgd_lr}")

    def _run_pgd(self, cand, target_node, budget, n, init):
        """One projected-ascent run from `init`; returns (mask, final_loss)."""
        s = init.clone()
        m = torch.zeros_like(s); v = torch.zeros_like(s)   # Adam moments
        for t in range(1, self.pgd_steps + 1):
            s = s.detach().requires_grad_(True)
            cand_vals = torch.cat([s, s])
            out = self._forward_cand(cand, cand_vals)
            loss = self._target_embedding_loss(out, target_node)   # maximize this (demotion)
            (g,) = torch.autograd.grad(loss, s)
            with torch.no_grad():
                if self.pgd_optim == "adam":
                    m = self.adam_b1 * m + (1 - self.adam_b1) * g
                    v = self.adam_b2 * v + (1 - self.adam_b2) * g * g
                    mhat = m / (1 - self.adam_b1 ** t)
                    vhat = v / (1 - self.adam_b2 ** t)
                    step = mhat / (vhat.sqrt() + 1e-8)
                else:
                    step = g
                s = self._project_box_budget(s + self.pgd_lr * step, budget)   # ASCENT
        with torch.no_grad():
            final_loss = float(self._target_embedding_loss(
                self._forward_cand(cand, torch.cat([s, s])), target_node))
        return s.detach().float(), final_loss

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        if target_node in self._pgd_mask_cache:
            return self._pgd_mask_cache[target_node]
        self._activate_surrogate_run(0)
        n = self._n
        budget = int(getattr(self.system_config.general, "budget", 5))
        j = torch.arange(n, device=self.device)
        tcol = torch.full((n,), target_node, device=self.device, dtype=torch.long)
        cand = torch.cat([torch.stack([tcol, j]), torch.stack([j, tcol])], dim=1)  # [2, 2N]
        base_init = torch.full((n,), budget / max(n, 1), device=self.device)
        best_mask, best_loss = None, -float("inf")
        gen = torch.Generator(device=self.device).manual_seed(int(self.seed) + target_node)
        for r in range(self.pgd_restarts):
            if r == 0:
                init = base_init                                   # uniform (= CLEAVE-PGD init)
            else:
                noise = torch.rand(n, generator=gen, device=self.device) * (budget / max(n, 1))
                init = self._project_box_budget(base_init + noise, budget)
            mask, loss = self._run_pgd(cand, target_node, budget, n, init)
            if loss > best_loss:
                best_loss, best_mask = loss, mask
        self._pgd_mask_cache[target_node] = best_mask
        return best_mask


ATTACKERS = {"cleave_pgd_pp": CLEAVEPGDppAttack}
