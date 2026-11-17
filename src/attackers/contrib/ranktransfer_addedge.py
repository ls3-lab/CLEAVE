"""RankTransfer: CLEAVE contrastive surrogate + a LISTWISE SOFT-RANK demotion loss.

Motivation (brainstorm Tier A / learning family): position_change is a RANK quantity, but every
gradient attacker optimizes 1-cos(z_t, q_t) on the target embedding alone. Optimizing cosine spends
the surrogate's limited fidelity on cosine accuracy we don't need; optimizing a smooth surrogate of
the actual RANK only needs the surrogate and victim to agree on WHICH competitors sit near q_t (the
"wall") -- which two low-pass GCNs largely do. So this attacks the surrogate transfer ceiling at the
OBJECTIVE level, with no change to CLEAVE's surrogate or candidate-gradient machinery.

Only _target_embedding_loss changes. CLEAVE's one-shot score_candidates backprops through it, so the
soft-rank objective propagates automatically.

L_rank(t) = mean_n sigma( (cos(z_n^clean, q_t) - cos(z_t, q_t)) / tau )
          = smooth count of competitors n that OUTRANK t against q_t.
Maximizing L_rank (the attack ascends the demotion signal) pushes cos(z_t,q_t) below the competitor
mass -> t drops in rank. Competitor embeddings z_n^clean are the surrogate's frozen clean embeddings
(cached); only z_t is a function of the candidate edge, so the gradient flows correctly through out[t].
For scale we use the top-K nearest competitors to q_t (the wall), not all N.
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.cleave_attack import CLEAVEAttack
from src.attackers import register_attacker
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class RankTransferAttack(CLEAVEAttack):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.rank_tau = float(getattr(cfg, "rank_tau", 0.05))
        self.rank_topk = int(getattr(cfg, "rank_topk", 512))    # competitor "wall" size
        self._wall_cache = {}
        logger.info(f"RankTransfer: listwise soft-rank loss (tau={self.rank_tau}, topK={self.rank_topk})")

    def _wall(self, target_node):
        """Top-K clean competitors nearest q_t (the rank wall), cached per target/surrogate."""
        key = (self._active_surrogate_run, target_node)
        w = self._wall_cache.get(key)
        if w is None:
            q = self._centroid_query(target_node)                       # [1,d] frozen query
            zc = self._clean_emb_full_cache[self._active_surrogate_run]  # [N,d] clean surrogate emb
            sims = F.cosine_similarity(zc, q, dim=1)                     # [N]
            k = min(self.rank_topk, zc.size(0))
            idx = torch.topk(sims, k).indices
            w = (zc[idx].detach(), q.detach())
            self._wall_cache[key] = w
        return w

    def _target_embedding_loss(self, out, target_node):
        zc_wall, q = self._wall(target_node)
        z_t = out[target_node:target_node + 1]                          # [1,d], depends on candidate edge
        cos_t = F.cosine_similarity(z_t, q, dim=1)                      # [1]
        cos_wall = F.cosine_similarity(zc_wall, q, dim=1)              # [K]
        # smooth count of competitors outranking t; higher when t is pushed below the wall
        return torch.sigmoid((cos_wall - cos_t) / self.rank_tau).mean()


register_attacker("ranktransfer", RankTransferAttack)
ATTACKERS = {"ranktransfer": RankTransferAttack}
