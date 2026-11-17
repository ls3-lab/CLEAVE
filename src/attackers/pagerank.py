from typing import Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from omegaconf import DictConfig

from src.attackers.base import Attacker


class PageRankAttack(Attacker):
    """
    PageRank-based attack that prioritizes removing edges to HIGH personalized-PageRank
    neighbors (the neighbors most connected to the target under a random walk personalized
    at it). The neighbor's PPR is the score and ``sort_reverse=True`` removes the highest-PPR
    neighbors first. (Removing low-PPR neighbors first — the least-connected ones — was found
    counterproductive under the rank-demotion metric: a weak choice that PROMOTED targets on
    some datasets, like the analogous low-degree bug in DegreeAttack.)
    """

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        super().__init__(cfg, system_config, data)
        self.sort_reverse = True  # remove highest-PageRank neighbors first

    def _compute_pagerank(self, target_node: int) -> dict:
        """Compute PageRank scores for all nodes."""
        G = to_networkx(self.data.to('cpu'), to_undirected=True)
        personalization = {target_node: 1}
        # personalization = {int(node): 1 for node in selected_nodes.cpu().tolist()}
        return nx.pagerank(G, personalization=personalization)

    def score_edges(self, target_node: int) -> dict[Tuple[int, int], float]:
        """
        Score each edge connected to target node based on neighbor PageRank.
        
        Returns:
            Dict mapping (src, dst) -> score (= neighbor personalized PageRank).
            Higher score = higher removal priority (sort_reverse=True => high-PageRank first).
        """
        target_edges = self.get_target_edges(target_node)
        scores = {}
        pagerank_scores = self._compute_pagerank(target_node)
        for edge in target_edges:
            neighbor = self.get_edge_neighbor(edge, target_node)
            scores[edge] = pagerank_scores.get(neighbor, 0.0)
        
        return scores

    def score_candidates(self, target_node: int) -> torch.Tensor:
        """Add-edge mode: attach the target to the HIGHEST personalized-PageRank
        nodes (those a random walk personalized at the target reaches most).

        Reuses ``_compute_pagerank`` (the exact removal signal) over ALL nodes.
        """
        num_nodes = int(getattr(self.data, "num_nodes", 0) or self.data.x.size(0))
        pr = self._compute_pagerank(target_node)
        vec = torch.zeros(num_nodes, dtype=torch.float, device=self.data.edge_index.device)
        for n, v in pr.items():
            vec[int(n)] = float(v)
        return vec
