"""CLEAVE-PGD: CLEAVE\'s contrastive surrogate + PGD multi-step joint edge optimization.

CLEAVE-Add (the plain version) scores every candidate non-edge with ONE gradient of the
demotion loss, then adds the top-budget. That is a one-shot greedy choice. PGD/Topology
Attack wins the add-edge benchmark because it instead optimizes a CONTINUOUS weight on all
candidate edges JOINTLY over many projected-gradient-ascent steps, capturing edge-set
interactions the one-shot method misses.

CLEAVE-PGD gives CLEAVE that same optimizer: it keeps CLEAVE\'s from-scratch self-supervised
contrastive surrogate (victim-free, strictly-less-access) but selects edges with PGD-style
multi-step projected ascent on L = 1 - cos(z_t, q_t) through that surrogate. So it is the
PGD attacker\'s optimizer with CLEAVE\'s (better-transferring, lower-access) surrogate.

Only score_candidates changes; the base _attack_add still rounds the optimized continuous
mask to the top-budget edges (exactly PGD\'s rounding step).
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.cleave_attack import CLEAVEAttack
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class CLEAVEPGDAttack(CLEAVEAttack):
    """CLEAVE with PGD/Topology multi-step joint edge optimization instead of one-shot gradient."""

    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.pgd_steps = int(getattr(cfg, "pgd_steps", 100))
        self.pgd_lr = float(getattr(cfg, "pgd_lr", 0.1))
        self._pgd_mask_cache = {}   # target -> optimized continuous mask [N] (built at max budget)
        logger.info(f"CLEAVE-PGD: contrastive surrogate + PGD ({self.pgd_steps} steps, lr={self.pgd_lr})")

    def _project_box_budget(self, s, budget):
        """Euclidean projection onto {0<=s<=1, sum(s)<=budget} via bisection on the dual."""
        s = s.clamp(0, 1)
        if float(s.sum()) <= budget:
            return s
        lo, hi = 0.0, float(s.max())
        for _ in range(50):
            mu = (lo + hi) / 2
            if float((s - mu).clamp(0, 1).sum()) > budget:
                lo = mu
            else:
                hi = mu
        return (s - hi).clamp(0, 1)

    def _forward_cand(self, cand_edges, cand_vals):
        """CLEAVE\'s sparse normalized-GCN forward, augmented with candidate directed edges
        weighted by cand_vals (differentiable). Mirrors AGPOA\'s augmented forward but uses
        CLEAVE\'s contrastive-surrogate weights (self._xW1, _W2, _b1, _b2)."""
        n = self._n
        E = self._edge_index.shape[1]
        all_idx = torch.cat([self._edge_index, cand_edges, self._self_loop_indices], dim=1)
        base = torch.ones(E, device=self.device)
        loops = torch.ones(n, device=self.device)
        vals = torch.cat([base, cand_vals, loops])
        adj = torch.sparse_coo_tensor(all_idx, vals, (n, n))
        deg = torch.sparse.mm(adj, torch.ones(n, 1, device=self.device)).squeeze(1)
        dis = deg.pow(-0.5)
        dis = torch.where(torch.isinf(dis), torch.zeros_like(dis), dis)
        row, col = all_idx
        norm = dis[row] * vals * dis[col]
        adj_norm = torch.sparse_coo_tensor(all_idx, norm, (n, n))
        h = torch.sparse.mm(adj_norm, self._xW1)
        if self._b1 is not None:
            h = h + self._b1
        h = F.relu(h)
        out = torch.sparse.mm(adj_norm, h @ self._W2.t())
        if self._b2 is not None:
            out = out + self._b2
        return out

    def score_candidates(self, target_node):
        """PGD multi-step projected ascent on the demotion loss over ALL candidate non-edges,
        through CLEAVE\'s contrastive surrogate. Optimized once per target at the max budget
        and cached; the base _attack_add rounds to the top-budget edges."""
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
        s = torch.full((n,), budget / max(n, 1), device=self.device)
        for _ in range(self.pgd_steps):
            s = s.detach().requires_grad_(True)
            cand_vals = torch.cat([s, s])                          # (t,j) and (j,t) share the weight
            out = self._forward_cand(cand, cand_vals)
            loss = self._target_embedding_loss(out, target_node)   # 1 - cos(z_t, q_t) via CLEAVE query
            (g,) = torch.autograd.grad(loss, s)
            s = self._project_box_budget(s + self.pgd_lr * g, budget)   # ASCENT: maximize demotion
        mask = s.detach().float()
        self._pgd_mask_cache[target_node] = mask
        return mask


ATTACKERS = {"cleave_pgd": CLEAVEPGDAttack}
