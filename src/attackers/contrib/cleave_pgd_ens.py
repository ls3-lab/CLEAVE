"""CLEAVE-PGD-Ensemble: diagnostic variant — PGD multi-step joint optimization whose
mask-gradient is AVERAGED across K from-scratch contrastive surrogates (seeds 42..42+K-1).

Identical to CLEAVEPGDAttack except that each PGD ascent step averages the loss-gradient
over the n_surrogates surrogates (L1-normalized per surrogate, equal-weight consensus, exactly
as CLEAVE's one-shot _compute_adj_gradient ensembles). This tests lever 1 (ensemble surrogate)
against the single-surrogate CLEAVE-PGD: same optimizer, same access model (still query-free /
victim-free), only the surrogate-variance is reduced. Set n_surrogates=1 to reproduce plain
CLEAVE-PGD exactly.
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.cleave_pgd import CLEAVEPGDAttack
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class CLEAVEPGDEnsembleAttack(CLEAVEPGDAttack):
    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        if target_node in self._pgd_mask_cache:
            return self._pgd_mask_cache[target_node]
        K = int(getattr(self, "n_surrogates", 1))
        n = self._n
        budget = int(getattr(self.system_config.general, "budget", 5))
        j = torch.arange(n, device=self.device)
        tcol = torch.full((n,), target_node, device=self.device, dtype=torch.long)
        cand = torch.cat([torch.stack([tcol, j]), torch.stack([j, tcol])], dim=1)  # [2, 2N]
        s = torch.full((n,), budget / max(n, 1), device=self.device)
        for _ in range(self.pgd_steps):
            s = s.detach().requires_grad_(True)
            cand_vals = torch.cat([s, s])
            # average the demotion-loss gradient across the K surrogates (equal-weight consensus)
            total_g = None
            for k in range(K):
                self._use_surrogate(k)                 # swaps _xW1/_W2/_b1/_b2 + active-run query key
                out = self._forward_cand(cand, cand_vals)
                loss = self._target_embedding_loss(out, target_node)
                (g,) = torch.autograd.grad(loss, s, retain_graph=(k < K - 1))
                g = g / (g.abs().sum() + 1e-12)        # L1-normalize per surrogate
                total_g = g if total_g is None else total_g + g
            self._use_surrogate(0)
            g = total_g / K
            s = self._project_box_budget(s + self.pgd_lr * g, budget)
        mask = s.detach().float()
        self._pgd_mask_cache[target_node] = mask
        return mask


ATTACKERS = {"cleave_pgd_ens": CLEAVEPGDEnsembleAttack}
