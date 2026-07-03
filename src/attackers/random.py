import random
from typing import Tuple
from torch_geometric.data import Data
from omegaconf import DictConfig

from src.attackers.base import Attacker


class RandomAttack(Attacker):
    """
    Random attack that assigns random scores to edges.
    """

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        super().__init__(cfg, system_config, data)

    def score_edges(self, target_node: int) -> dict[Tuple[int, int], float]:
        """
        Score each edge connected to target node randomly.
        
        Returns:
            Dict mapping (src, dst) -> score
            Random scores for random removal order
        """
        target_edges = self.get_target_edges(target_node)
        scores = {}
        
        for edge in target_edges:
            # Assign random score
            scores[edge] = random.random()
        
        return scores