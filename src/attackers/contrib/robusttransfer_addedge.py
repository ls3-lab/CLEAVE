"""RobustTransfer: vote across a diverse surrogate population instead of averaging their gradients.

The earlier finding was that surrogate ENSEMBLING (averaging gradients / embeddings) HURT. The brainstorm
diagnosis: averaging raw gradients superimposes disagreeing per-surrogate directions and collapses toward
a safe, low-magnitude (often promotion-ish) consensus. The fix is to aggregate at the DECISION level:
score each candidate edge by its mean RANK across the surrogate population's individual demotion gradients,
so an edge must be agreed-strong by MANY surrogates to win -- selecting for family-level (low-pass GCN)
signal while discarding any single surrogate's idiosyncratic direction.

Reuses CLEAVE's n_surrogates training + per-run one-shot candidate gradient (score_candidates on run k).
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.contrib.cleave_attack import CLEAVEAttack
from src.attackers import register_attacker
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class RobustTransferAttack(CLEAVEAttack):
    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        super().__init__(cfg, system_config, data, retriever=retriever, **kwargs)
        self.vote_mode = str(getattr(cfg, "vote_mode", "rank"))   # rank | min (minimax)
        if self.n_surrogates < 2:
            logger.warning(f"RobustTransfer with n_surrogates={self.n_surrogates}: needs >=2 to vote; "
                           "falls back to plain CLEAVE gradient.")
        logger.info(f"RobustTransfer: {self.n_surrogates}-surrogate {self.vote_mode}-vote candidate selection")

    def _activate_surrogate_run(self, k):
        # The inherited add-edge score_candidates hardcodes run 0 at its top; when a pin is set we
        # remap that to the surrogate we actually want to score under, so the vote uses all K.
        pin = getattr(self, "_pinned_run", None)
        return super()._activate_surrogate_run(pin if pin is not None else k)

    def score_candidates(self, target_node):
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        if self.n_surrogates < 2:
            return super().score_candidates(target_node)
        per = []
        for k in range(self.n_surrogates):
            self._pinned_run = k
            per.append(super().score_candidates(target_node).detach().float())  # scored under surrogate k
        self._pinned_run = None
        self._activate_surrogate_run(0)
        S = torch.stack(per, 0)                          # [K, N]
        if self.vote_mode == "min":
            return S.min(0).values                        # minimax: agreed by the worst surrogate
        # rank-vote: mean rank position (higher score -> better rank) across surrogates
        ranks = torch.argsort(torch.argsort(S, dim=1), dim=1).float()  # [K,N] rank of each node per run
        return ranks.mean(0)                              # [N] mean rank = consensus strength


register_attacker("robusttransfer", RobustTransferAttack)
ATTACKERS = {"robusttransfer": RobustTransferAttack}
