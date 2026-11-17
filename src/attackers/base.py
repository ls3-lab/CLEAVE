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
        Execute the attack for one target node.

        When ``add_edge`` is False (default) the attack REMOVES the top-``budget``
        scored undirected edges incident to the target (the original behaviour).
        When ``add_edge`` is True it ADDS the top-``budget`` scored non-edges
        {target, j}, using the attacker's :meth:`score_candidates` hook — see
        :meth:`_attack_add`.

        Returns a list of modified graphs (one per trial).
        """
        if add_edge:
            return self._attack_add(target_node, budget)

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

    # ------------------------------------------------------------------ #
    # Add-edge mode (shared machinery for every attacker that supports it)
    # ------------------------------------------------------------------ #
    def score_candidates(self, target_node: int) -> "torch.Tensor":
        """Score every node as a candidate NEW neighbor of ``target_node``.

        Returns a length-``num_nodes`` float tensor; a HIGHER score means the
        non-edge ``{target_node, j}`` is added first. The base masks out the
        target itself and its current neighbors before selecting, so a subclass
        may return raw scores over all nodes.

        Attackers that support add-edge mode override this. The default raises,
        so flipping ``add_edge: true`` on an attacker that has not been adapted
        fails loudly instead of silently doing nothing.
        """
        raise NotImplementedError(
            f"Attacker '{self.name}' does not support add-edge mode "
            f"(no score_candidates implementation)."
        )

    def _add_candidate_mask(self, target_node: int) -> "torch.Tensor":
        """Boolean [N] mask: True for positions that are NOT valid new-edge
        endpoints (the target itself and its current neighbors)."""
        ei = self.data.edge_index
        num_nodes = int(getattr(self.data, "num_nodes", 0) or self.data.x.size(0))
        nbrs = torch.cat([ei[1, ei[0] == target_node], ei[0, ei[1] == target_node]])
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=ei.device)
        if nbrs.numel() > 0:
            mask[nbrs] = True
        mask[target_node] = True
        return mask

    def _attack_add(self, target_node: int, budget: int) -> List[Data]:
        """Insert the top-``budget`` scored non-edges incident to the target.

        GPU-resident selection: ``score_candidates`` returns an [N] tensor on
        the graph device, invalid endpoints are masked to -inf, and
        ``torch.topk`` picks the budget best on-device. Each trial reseeds so
        stochastic scorers (e.g. random) vary.
        """
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        out = []
        for i in range(self.trials):
            torch.manual_seed(self.seed + i)
            torch.cuda.manual_seed(self.seed + i)
            random.seed(self.seed + i)
            np.random.seed(self.seed + i)
            scores = self.score_candidates(target_node).clone().float()
            scores[self._add_candidate_mask(target_node)] = float("-inf")
            finite = int(torch.isfinite(scores).sum().item())
            k = min(budget, finite)
            if k <= 0:
                out.append(self._add_edge([]))
                continue
            top = torch.topk(scores, k).indices.tolist()
            edges = [(int(target_node), int(j)) for j in top]
            out.append(self._add_edge(edges))
        return out

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
        Add the given undirected edges to the graph.

        Symmetric standard (mirrors ``_remove_edges``): each ``(a, b)`` is
        undirected, so **both** ``(a, b)`` and ``(b, a)`` directed entries are
        inserted, keeping the adjacency symmetric (the GCN retrievers assume a
        symmetric adjacency). ``budget`` therefore counts undirected edges
        (= 2x directed entries). Edges already present in the graph (or
        self-loops) are skipped, so the realized perturbation never silently
        duplicates an existing edge.

        Sets ``edges_added`` (the directed entries actually inserted) and
        ``edges_removed = []`` so the evaluator's ``_removed_edges`` helper
        short-circuits (an add-edge attack removes nothing).
        """
        if len(edges) == 0:
            logger.warning('No edges to add in attack')
            updated_data = self.data.clone()
            updated_data.edges_added = []
            updated_data.edges_removed = []
            return updated_data

        dev = self.data.edge_index.device
        num_nodes = int(getattr(self.data, "num_nodes", 0) or self.data.x.size(0))

        # Existing directed entries encoded as int64 keys (a * N + b), computed
        # ONCE per attacker instance and cached on the GPU — avoids rebuilding a
        # python set of every edge on the CPU on every _add_edge call (the
        # per-call O(E) bottleneck on large graphs like PubMed).
        if getattr(self, "_existing_edge_keys", None) is None or self._existing_edge_keys.device != dev:
            ei = self.data.edge_index
            self._existing_edge_keys = (ei[0].to(torch.long) * num_nodes + ei[1].to(torch.long)).to(dev)

        # Expand each undirected edge to both directed entries on-device; skip
        # self-loops and edges already present (GPU set-membership via isin).
        cand = torch.tensor([(int(a), int(b)) for (a, b) in edges if int(a) != int(b)],
                            dtype=torch.long, device=dev)
        if cand.numel() == 0:
            updated_data = self.data.clone()
            updated_data.edges_added = []
            updated_data.edges_removed = []
            return updated_data
        both = torch.cat([cand, cand.flip(1)], dim=0)            # both directions
        both = torch.unique(both, dim=0)
        keys = both[:, 0] * num_nodes + both[:, 1]
        keep = ~torch.isin(keys, self._existing_edge_keys)       # drop already-present
        both = both[keep]

        updated_data = self.data.clone()
        if both.size(0) == 0:
            logger.warning('No new edges to add (all requested edges already present)')
            updated_data.edges_added = []
            updated_data.edges_removed = []
            return updated_data

        updated_data.edge_index = torch.cat([self.data.edge_index, both.t()], dim=1)
        updated_data.edges_added = [tuple(map(int, e)) for e in both.tolist()]
        updated_data.edges_removed = []

        return updated_data
