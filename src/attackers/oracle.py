import os
import pickle
from typing import Tuple

from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.oracle.cache import oracle_cache_filename
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class OracleAttack(Attacker):
    """Oracle attack.

    Removes the precomputed OPTIMAL set of incident undirected edges for the requested
    budget — the synergy-aware best edge *set* (exhaustive over the target's top-K most
    impactful incident edges), not a greedy incremental sequence. The cache is built by
    ``build_oracle.py`` for the attack target nodes:
        {node: {"init_position": int, "optimal": {budget: {"edges": [(u,v)...], ...}}}}
    """

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        self.cfg = cfg
        self.system_config = system_config
        super().__init__(cfg, system_config, data)
        self.sort_reverse = True
        self.oracle = self._load_oracle()

    def _default_path(self) -> str:
        g = self.system_config.general
        cutoff = int(self.system_config.dataset.brute_cutoff)
        beam_width = int(getattr(g, "oracle_beam_width", 100))
        return os.path.join(
            g.cache_dir, "oracle",
            oracle_cache_filename(self.system_config.retriever.name.upper(), g.budget,
                                  cutoff, beam_width, self.system_config.dataset.name, g.graph_model),
        )

    def _load_oracle(self) -> dict:
        path = getattr(self.cfg, "oracle_cache_path", None) or self._default_path()
        logger.info(f"Loading oracle cache: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Oracle cache not found: {path}\nGenerate it first, e.g.:\n"
                f"  python build_oracle.py dataset={self.system_config.dataset.name} "
                f"retriever={self.system_config.retriever.name} +general.force_rewrite=true\n"
                f"(or set attacker.oracle_cache_path to an existing cache)."
            )
        with open(path, "rb") as f:
            oracle = pickle.load(f)
        logger.info(f"Loaded oracle for {len(oracle)} target nodes")
        return oracle

    @staticmethod
    def _to_int(node) -> int:
        return int(node.item()) if hasattr(node, "item") else int(node)

    def score_edges(self, target_node: int) -> dict[Tuple[int, int], float]:
        """Fallback scoring (the budget-max optimal set). attack() is the primary path."""
        rec = self.oracle.get(self._to_int(target_node))
        if not rec or not rec.get("optimal"):
            return {}
        edges = rec["optimal"][max(rec["optimal"].keys())]["edges"]
        return {tuple(map(int, e)): float(len(edges) - i) for i, e in enumerate(edges)}

    def attack(self, target_node: int, budget: int) -> list[Data]:
        node = self._to_int(target_node)
        rec = self.oracle.get(node)
        if not rec or not rec.get("optimal"):
            return [self._remove_edges([])]  # not a target node / no edges
        optimal = rec["optimal"]
        b = budget if budget in optimal else min(max(optimal.keys()), budget)
        edges = optimal.get(b, optimal[max(optimal.keys())])["edges"]
        return [self._remove_edges(edges)]
