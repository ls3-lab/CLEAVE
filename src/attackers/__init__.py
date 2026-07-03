from typing import List
import importlib
import pkgutil

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.retrievers.base import Retriever
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# --- Baseline / oracle / surrogate attackers shipped with the reference repo ---
from src.attackers.oracle import OracleAttack
from src.attackers.degree import DegreeAttack
from src.attackers.random import RandomAttack
from src.attackers.pagerank import PageRankAttack
from src.attackers.rlp import RLPAttack
from src.attackers.viking import VikingAttack
from src.attackers.infattack import InfAttack
from src.attackers.infmax import InfMaxAttack
from src.attackers.graphmi import GraphMIAttackRetriever
from src.attackers.graphzoom import GraphZOOMAttack
from src.attackers.agpoa import AGPOAAttackRetriever
from src.attackers.nettack import NettackAttackRetriever
from src.attackers.nag_r import NAGRAttackRetriever

logger = setup_logger(__name__)

__all__ = [
    'Attacker',
    'DegreeAttack',
    'RandomAttack',
    'PageRankAttack',
    'RLPAttack',
    'OracleAttack',
    'VikingAttack',
    'InfAttack',
    'InfMaxAttack',
    'GraphZOOMAttack',
    'AGPOAAttackRetriever',
    'NettackAttackRetriever',
    'NAGRAttackRetriever',
    'GraphMIAttackRetriever',
    'register_attacker',
    'load_attackers',
    'ATTACKERS_REGISTRY',
]

# Built-in attackers. Keys are the names used in conf/attacker/*.yaml.
ATTACKERS_REGISTRY = {
    "oracle": OracleAttack,
    "degree": DegreeAttack,
    "pagerank": PageRankAttack,
    "infattack": InfAttack,        # back-compat alias (defaults to label_mode=kl)
    "infattack_kl": InfAttack,     # Known-Labels: ground-truth target/neighbor labels
    "infattack_ul": InfAttack,     # Unknown-Labels: surrogate-predicted labels (no GT for targets)
    "infmax": InfMaxAttack,
    "rlp": RLPAttack,
    "random": RandomAttack,
    "viking": VikingAttack,
    "graphzoom": GraphZOOMAttack,
    "graphmi": GraphMIAttackRetriever,
    # Tier-1 gradient attackers, two drift objectives (same classes; drift_reference query|self):
    #   query-drift = drift the target from the centroid query the metric ranks against (strong, aligned)
    #   self-drift  = drift the target from its OWN clean embedding (query-free ablation; weaker)
    "agpoa_querydrift": AGPOAAttackRetriever,
    "agpoa_selfdrift": AGPOAAttackRetriever,
    "nettack_querydrift": NettackAttackRetriever,
    "nettack_selfdrift": NettackAttackRetriever,
    "nag_r_querydrift": NAGRAttackRetriever,
    "nag_r_selfdrift": NAGRAttackRetriever,
}


def register_attacker(name: str, attacker_cls) -> None:
    """Register an attacker class under ``name``.

    Forks add their own (proposed) methods without editing this tracked file:
    drop a module under ``src/attackers/contrib/`` (gitignored) that either
    exposes an ``ATTACKERS = {name: cls}`` dict or calls ``register_attacker``
    at import time. See ``_load_contrib_attackers``.
    """
    ATTACKERS_REGISTRY[name] = attacker_cls


def _load_contrib_attackers() -> None:
    """Optionally auto-discover attackers under the gitignored ``contrib`` package.

    This keeps researcher-specific / proposed methods out of the reference repo
    while letting a fork register them with zero edits to tracked code. A missing
    ``contrib`` package is the normal case and is silently ignored.
    """
    try:
        import src.attackers.contrib as contrib_pkg  # noqa: WPS433
    except ImportError:
        return
    for mod in pkgutil.iter_modules(contrib_pkg.__path__):
        try:
            module = importlib.import_module(f"src.attackers.contrib.{mod.name}")
        except Exception as exc:  # don't let a broken contrib module break the framework
            logger.warning(f"Skipping contrib attacker module '{mod.name}': {exc}")
            continue
        for name, cls in getattr(module, "ATTACKERS", {}).items():
            register_attacker(name, cls)
            logger.info(f"Registered contrib attacker '{name}' from contrib.{mod.name}")


_load_contrib_attackers()


def load_attackers(attackers_configs: List[DictConfig],
                   system_config: DictConfig,
                   data: Data,
                   device: str,
                   seed: int,
                   retriever: Retriever,
                   victim_retriever: Retriever = None) -> List[Attacker]:
    # Black-box transfer (default): attackers distill/optimize on the SURROGATE retriever (`retriever`);
    # the VICTIM is only the evaluation target. An attacker whose published threat model is white-box may
    # opt into the victim via `use_victim_retriever: true` in its config (e.g. graphmi) — reported
    # separately, not head-to-head with the surrogate-transfer baselines.
    attackers = []
    for attacker_config in attackers_configs:
        atk_retriever = retriever
        if getattr(attacker_config, "use_victim_retriever", False) and victim_retriever is not None:
            atk_retriever = victim_retriever
        try:
            attacker = ATTACKERS_REGISTRY[attacker_config.name](
                attacker_config,
                system_config=system_config,
                data=data,
                device=device,
                seed=seed,
                retriever=atk_retriever,
            )
        except Exception as e:
            # One attacker failing to construct (e.g. an OOM inside a heavy
            # attacker's __init__ on a large graph) must not abort the whole
            # (retriever, dataset) run — log it and skip just that attacker.
            logger.warning(
                "[skip-construct] attacker '%s' failed to initialize "
                "(%s: %s); skipping it for this run.",
                attacker_config.name, type(e).__name__, e,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        attackers.append(attacker)
    return attackers
