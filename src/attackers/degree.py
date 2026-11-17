import torch
from typing import Tuple
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src.attackers.base import Attacker


class DegreeAttack(Attacker):
    """
    Degree-based attack that prioritizes removing edges to HIGH-degree neighbors,
    matching the config ("removes edges connected to high-degree nodes"). The
    neighbor's degree is the score and ``sort_reverse=True`` removes the highest-
    degree neighbors first. (The earlier low-degree-first variant was found to be
    counterproductive under the rank-demotion metric — it PROMOTED the target,
    giving negative position_change at low budgets.)
    """

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        super().__init__(cfg, system_config, data)
        self.degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
        self.sort_reverse = True  # remove highest-degree neighbors first

    def score_edges(self, target_node: int) -> dict[Tuple[int, int], float]:
        """
        Score each edge connected to target node by the neighbor's degree.

        Returns:
            Dict mapping (src, dst) -> score (= neighbor degree).
            Higher score = higher removal priority (sort_reverse=True => high-degree first).
        """
        target_edges = self.get_target_edges(target_node)
        scores = {}

        for edge in target_edges:
            neighbor = self.get_edge_neighbor(edge, target_node)
            scores[edge] = float(self.degrees[neighbor])

        return scores

    def score_candidates(self, target_node: int) -> torch.Tensor:
        """Add-edge mode: attach the target to the HIGHEST-degree nodes (hubs).

        Mirrors the removal signal (neighbor degree). ``sort_reverse`` is unused
        in add-edge mode; the base always takes the top-scored (= highest-degree)
        candidates.
        """
        return self.degrees.to(self.data.edge_index.device).float()
