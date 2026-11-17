import random
from typing import Tuple
import torch
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

    def score_candidates(self, target_node: int) -> torch.Tensor:
        """Add-edge mode: uniform-random score over all candidate new neighbors."""
        num_nodes = int(getattr(self.data, "num_nodes", 0) or self.data.x.size(0))
        return torch.rand(num_nodes, device=self.data.edge_index.device)
