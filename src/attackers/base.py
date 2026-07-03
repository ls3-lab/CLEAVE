"""Abstract base class for attackers.

Undirected, symmetric standard: all supported datasets are undirected graphs
(``edge_index`` stores both ``(u, v)`` and ``(v, u)``). An "edge" here is the
undirected pair ``{target, neighbor}``; the **budget counts undirected edges**,
and removing one edge removes **both** directed entries so the graph stays
symmetric (the GCN retrievers assume a symmetric adjacency).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from omegaconf import DictConfig
import torch
from torch_geometric.data import Data
from src.utils.logger_utils import setup_logger
import random
import numpy as np

logger = setup_logger(__name__)


class Attacker(ABC):
    """
    Base class for edge-removal attacks.

    An attacker selects which (undirected) edges incident to a target node to
    remove in order to demote the node in retrieval. A budget of ``k`` removes
    ``k`` undirected edges (= ``2k`` directed ``edge_index`` entries).
    """

    def __init__(self, cfg, system_config: DictConfig, data: Data):
        """
        Args:
            cfg: Configuration object
            system_config: System configuration object
            data: Graph data
        """
        self.cfg = cfg
        self.system_config = system_config
        logger.info(cfg)
        self.trials = cfg.trials if hasattr(cfg, 'trials') else 1
        self.seed = system_config.general.seed
        self.name = cfg.name
        self.data = data
        self.sort_reverse = False
        self.device = data.edge_index.device
        self.edge_direction = self._get_edge_direction()
        logger.info(
            f'Attackers: {self.name}, seed: {self.seed}, trials: {self.trials}, '
            f'edge_direction: {self.edge_direction}'
        )

    def _get_edge_direction(self) -> str:
        """Undirected standard: edges are symmetric, so direction is always ``both``.

        Datasets are undirected and edges are removed symmetrically, so the
        incoming/outgoing distinction is meaningless. A configured non-``both``
        value is ignored (with a warning) for backward compatibility.
        """
        direction = None
        if hasattr(self.cfg, "edge_direction"):
            direction = self.cfg.edge_direction
        elif hasattr(self.system_config.general, "edge_direction"):
            direction = self.system_config.general.edge_direction

        if direction is not None and str(direction).lower() not in ("both", "all", "none"):
            logger.warning(
                f"edge_direction={direction!r} is ignored: graphs are undirected and edges "
                f"are removed symmetrically ('both')."
            )
        return "both"

    @abstractmethod
    def score_edges(self, target_node: int) -> dict[Tuple[int, int], float]:
        """
        Score each undirected edge incident to target_node.

        Returns:
            Dict mapping ``(target_node, neighbor)`` -> score. Sorted ascending by
            default (lower score = higher removal priority); set
            ``self.sort_reverse = True`` to remove highest-scored edges first.
        """
        pass

    def get_target_edges(self, target_node: int) -> List[Tuple[int, int]]:
        """Return the undirected edges incident to target_node.

        One canonical ``(target_node, neighbor)`` tuple per unique neighbor — the
        two directed entries of an undirected edge collapse to a single edge.
        Self-loops are skipped.
        """
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        edge_index = self.data.edge_index
        mask = (edge_index[0] == target_node) | (edge_index[1] == target_node)
        cols = edge_index[:, mask].t().tolist()

        edges: List[Tuple[int, int]] = []
        seen = set()
        for src, dst in cols:
            src, dst = int(src), int(dst)
            neighbor = dst if src == target_node else src
            if neighbor == target_node:  # skip self-loops
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                edges.append((target_node, neighbor))
        return edges

    @staticmethod
    def get_edge_neighbor(edge: Tuple[int, int], target_node: int) -> int:
        """Return the endpoint of an incident edge that is not target_node."""
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        src, dst = int(edge[0]), int(edge[1])
        return dst if src == target_node else src

    def attack(self, target_node: int, budget: int, add_edge: bool = False) -> Data:
        """
        Execute attack by removing the top-``budget`` scored undirected edges.

        Returns:
            List of modified graphs (one per trial), each with ``budget``
            undirected edges removed symmetrically.
        """
        embedding_list = []
        for i in range(self.trials):
            torch.manual_seed(self.seed + i)
            torch.cuda.manual_seed(self.seed + i)
            random.seed(self.seed + i)
            np.random.seed(self.seed + i)
            scores = self.score_edges(target_node)

            # Sort by score and take the top `budget` undirected edges.
            sorted_edges = sorted(scores.keys(), key=lambda e: scores[e], reverse=self.sort_reverse)
            edges_to_remove = sorted_edges[:budget]

            embedding_list.append(self._remove_edges(edges_to_remove))
        return embedding_list

    def _remove_edges(self, edges: List[Tuple[int, int]]) -> Data:
        """
        Remove the given undirected edges from the graph.

        Each ``(a, b)`` is treated as undirected: **both** ``(a, b)`` and
        ``(b, a)`` directed entries are removed, so the graph stays symmetric.
        ``budget`` therefore counts undirected edges (= 2x directed entries).
        """
        # Expand each undirected edge to both directed entries; dedup.
        directed = set()
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            directed.add((a, b))
            directed.add((b, a))

        if len(directed) == 0:
            logger.warning('No edges to remove in attack')
            return self.data.clone()

        edges_to_remove_tensor = torch.tensor(
            sorted(directed), dtype=torch.long, device=self.data.edge_index.device
        )

        edge_index_T = self.data.edge_index.t()  # shape [num_edges, 2]
        mask = ~((edge_index_T[:, None, :] == edges_to_remove_tensor[None, :, :]).all(dim=2).any(dim=1))

        updated_data = self.data.clone()
        updated_data.edge_index = self.data.edge_index[:, mask]
        updated_data.edges_removed = [tuple(map(int, edge)) for edge in edges_to_remove_tensor.tolist()]

        return updated_data

    def _add_edge(self, edges: List[Tuple[int, int]]) -> Data:
        """
        Add the edges to the data.
        """
        if len(edges) == 0:
            logger.warning('No edges to add in attack')
            return self.data.clone()

        edges_to_add = list(set(edges))
        edges_to_add_tensor = torch.tensor(edges_to_add, dtype=torch.long, device=self.data.edge_index.device).t()

        updated_data = self.data.clone()
        updated_data.edge_index = torch.cat([self.data.edge_index, edges_to_add_tensor], dim=1)

        return updated_data
