"""Build the oracle cache.

For each attack TARGET node, find the OPTIMAL set of incident undirected edges to remove
for each budget 1..B (synergy-aware — the best edge *set*, not greedy incremental).

- Candidates are the target's immediate (incident) undirected edges (the only edges any
  attacker may remove).
- DEGREE-TIERED search: targets with degree <= `dataset.brute_cutoff` are searched
  EXHAUSTIVELY over all incident edges; higher-degree targets fall back to a BEAM search of
  width `oracle_beam_width` (default 100) to keep the combinatorics tractable.
- Built only for the attack target nodes (same selection as retriever_attack.py), so the
  cache depends on the target-selection seed/ratio.
- Removal is undirected-symmetric (both directed entries), matching src/attackers/base.py.
- Writes cache/oracle/<RET>_Oracle_v2_Budget<B>_cut<cutoff>_W<W>_<Dataset>_<model>.pkl.

Run, e.g.:
  python build_oracle.py dataset=cora retriever=epagcl +general.force_rewrite=true
"""

import os
import sys
import pickle
import tempfile
import itertools

import hydra
import torch
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import get_dataset
from src.data.splits import load_or_create_attack_split
from src.retrievers import get_retriever
from src.oracle.local_encode import LocalOracleEngine
from src.oracle.query import neighborhood_centroid_query
from src.oracle.cache import oracle_cache_filename
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

def atomic_pickle_dump(obj, path):
    """Write a pickle atomically: dump to a temp file in the same dir, then os.replace.

    os.replace is atomic on POSIX, so a preemption mid-write can never truncate/corrupt an
    existing cache — the old file stays intact until the new one is fully written.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def shard_nodes(nodes, shard_index, shard_count):
    """Contiguous round-robin slice: node i goes to shard (i % shard_count). Deterministic,
    complete, non-overlapping. shard_count<=1 returns all nodes."""
    if shard_count <= 1:
        return list(nodes)
    return [n for i, n in enumerate(nodes) if i % shard_count == shard_index]


def merge_shard_caches(canonical_path):
    """Union all ``<canonical>.partN.pkl`` shard files into the canonical cache.

    Sharded array tasks each write their OWN part file (disjoint node sets), so concurrent runs
    never overwrite each other. This unions the parts into the single file the consumer reads.
    Returns (n_parts, n_nodes_merged).
    """
    import glob
    parts = sorted(glob.glob(f"{canonical_path[:-4]}.part*.pkl"))
    merged = {}
    for p in parts:
        with open(p, "rb") as f:
            merged.update(pickle.load(f))
    atomic_pickle_dump(merged, canonical_path)
    return len(parts), len(merged)


DEFAULT_TOPK = 15


def _cfg_get(cfg, key, default):
    return getattr(cfg.general, key) if hasattr(cfg.general, key) else default


def oracle_cache_path(cfg, save_path, brute_cutoff, beam_width):
    return os.path.join(
        save_path,
        oracle_cache_filename(cfg.retriever.name.upper(), cfg.general.budget,
                              brute_cutoff, beam_width, cfg.dataset.name, cfg.general.graph_model),
    )


def _incident_edges(data, node):
    """Canonical (node, neighbor) per unique neighbor (undirected); skip self-loops."""
    ei = data.edge_index
    mask = (ei[0] == node) | (ei[1] == node)
    cols = ei[:, mask].t().tolist()
    seen, out = set(), []
    for s, d in cols:
        s, d = int(s), int(d)
        nb = d if s == node else s
        if nb == node or nb in seen:
            continue
        seen.add(nb)
        out.append((node, nb))
    return out


def _remove_edges_from_data(data, edges):
    """Undirected-symmetric removal: drop both directed entries of each edge."""
    if not edges:
        return data
    directed = set()
    for a, b in edges:
        a, b = int(a), int(b)
        directed.add((a, b))
        directed.add((b, a))
    eit = data.edge_index.t()
    et = torch.tensor(sorted(directed), dtype=torch.long, device=data.edge_index.device)
    mask = ~((eit[:, None, :] == et[None, :, :]).all(dim=2).any(dim=1))
    updated = data.clone()
    updated.edge_index = data.edge_index[:, mask]
    return updated


def _global_cache(retriever, data):
    return {"dataset_embedding": retriever.encode(data.x, data.edge_index)}


def _node_query(retriever, data, node, global_cache, radius):
    """Frozen neighborhood-centroid query embedding + initial rank (mirrors evaluate_retrieval).
    Built from the surrogate's CLEAN full-graph embeddings (global_cache["dataset_embedding"])."""
    num_nodes = int(getattr(data, "num_nodes", 0) or data.x.size(0))
    q = neighborhood_centroid_query(data.edge_index, node, radius,
                                    global_cache["dataset_embedding"], num_nodes)
    ranking = retriever.retrieve(q, global_cache["dataset_embedding"])
    init = int((ranking == node).nonzero(as_tuple=True)[1].item())
    return q, init


def _position(retriever, attacked, node, q):
    emb = retriever.encode(attacked.x, attacked.edge_index)
    ranking = retriever.retrieve(q, emb)
    return int((ranking == node).nonzero(as_tuple=True)[1].item())


def compute_node_oracle(data, retriever, node, budget, topk, global_cache, radius,
                        candidate_strategy="topk"):
    """Optimal incident-edge set per budget for one target node (full-graph REFERENCE path).

    Semantics match the localized engine: optimal[b] = best demotion using 1..b edges (budget CAP;
    position_change MONOTONE in b). candidate_strategy:
      "topk"  -> exhaustive over the top-`topk` single-edge-ranked incident edges (the path the
                 golden gate compares against the localized engine in topk mode).
      "brute" -> exhaustive over ALL incident edges (the true optimum; only tractable on small
                 graphs / low degree — used as an independent reference in the tiered tests).
    Beam is NOT implemented here (the localized engine owns the production tiered search); this
    reference path is exact-by-enumeration only.
    """
    q, init = _node_query(retriever, data, node, global_cache, radius)
    candidates = _incident_edges(data, node)

    if candidate_strategy == "brute":
        pool = candidates
    elif candidate_strategy == "topk":
        scored = []
        for e in candidates:
            pos = _position(retriever, _remove_edges_from_data(data, [e]), node, q)
            scored.append((pos - init, e))
        scored.sort(key=lambda x: -x[0])
        pool = [e for _, e in scored[:topk]]
    else:
        raise ValueError(f"unknown candidate_strategy {candidate_strategy!r} (full path: topk | brute)")

    if len(pool) == 0:
        return {"init_position": init,
                "optimal": {b: {"edges": [], "position_change": 0, "final_position": init}
                            for b in range(1, budget + 1)}}

    optimal, run_pos, run_set = {}, -1, None
    for b in range(1, budget + 1):
        if b <= len(pool):                                  # best exact-size-b set
            best_pos, best_set = -1, None
            for combo in itertools.combinations(pool, b):
                pos = _position(retriever, _remove_edges_from_data(data, list(combo)), node, q)
                if pos > best_pos:
                    best_pos, best_set = pos, combo
            if best_pos > run_pos:                          # cumulative max -> <=b
                run_pos, run_set = best_pos, list(best_set)
        optimal[b] = {
            "edges": [tuple(map(int, e)) for e in run_set],
            "position_change": run_pos - init,
            "final_position": run_pos,
        }
    return {"init_position": init, "optimal": optimal}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.general.seed)
    logger.info(f"Working directory: {HydraConfig.get().runtime.output_dir}")
    device = cfg.general.device if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(cfg.dataset).to(device)
    data = dataset[0]

    budget = int(cfg.general.budget)
    topk = int(_cfg_get(cfg, "oracle_topk", DEFAULT_TOPK))
    radius = str(cfg.dataset.query_radius)                          # centroid query neighborhood
    brute_cutoff = int(cfg.dataset.brute_cutoff)                    # deg<=cutoff -> brute, else beam
    beam_width = int(_cfg_get(cfg, "oracle_beam_width", 100))
    use_victim = bool(_cfg_get(cfg, "oracle_uses_victim", False))  # default: build on the oracle model
    flush_every = int(_cfg_get(cfg, "oracle_flush_every", 25))
    force = bool(_cfg_get(cfg, "force_rewrite", False))

    # Oracle target scope (general.oracle_targets):
    #   "all"  (default) -> train ∪ test  (test = eval ceiling; train = proposed-method supervision)
    #   "test"           -> test only  (the eval ceiling the REFERENCE framework actually reports;
    #                       retriever_attack.py attacks only test targets, so this is sufficient and
    #                       ~5x cheaper — forks that train a proposed method rebuild with "all").
    train_ids, test_ids = load_or_create_attack_split(
        data, cfg.dataset.name, cfg.general.cache_dir, seed=cfg.general.seed)
    targets_mode = str(_cfg_get(cfg, "oracle_targets", "all"))
    if targets_mode == "test":
        targets = sorted(set(test_ids))
    elif targets_mode == "train":
        targets = sorted(set(train_ids))   # extend an existing test-only cache to train∪test (merge unions parts)
    else:
        targets = sorted(set(train_ids) | set(test_ids))
    shard_index = int(os.environ.get("ORACLE_SHARD_INDEX", _cfg_get(cfg, "oracle_shard_index", 0)))
    shard_count = int(os.environ.get("ORACLE_SHARD_COUNT", _cfg_get(cfg, "oracle_shard_count", 1)))
    targets = shard_nodes(targets, shard_index, shard_count)
    logger.info(f"Shard {shard_index}/{shard_count}: {len(targets)} target nodes (targets={targets_mode})")

    save_path = os.path.join(cfg.general.cache_dir, "oracle")
    os.makedirs(save_path, exist_ok=True)
    canonical_path = oracle_cache_path(cfg, save_path, brute_cutoff, beam_width)
    if bool(_cfg_get(cfg, "oracle_merge", False)):
        n_parts, n_nodes = merge_shard_caches(canonical_path)
        logger.info(f"Merged {n_parts} shard part(s) -> {canonical_path} ({n_nodes} nodes)")
        return
    # Each shard writes its OWN part file (union them afterward with +general.oracle_merge=true),
    # so concurrent array tasks can't overwrite each other's nodes via a shared cache file.
    file_path = canonical_path if shard_count <= 1 else f"{canonical_path[:-4]}.part{shard_index}.pkl"

    retriever = get_retriever(cfg.retriever, cfg, data, device, use_oracle=(not use_victim))
    oracle = {}
    if os.path.exists(file_path) and not force:
        with open(file_path, "rb") as f:
            oracle = pickle.load(f)
    todo = [int(n) for n in targets if force or int(n) not in oracle]
    logger.info(
        f"Oracle build: {len(todo)}/{len(targets)} target nodes | budget={budget} topk={topk} "
        f"query={radius} brute_cutoff={brute_cutoff} W={beam_width} "
        f"model={'victim' if use_victim else 'oracle'} -> {file_path}"
    )
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Default to the localized engine (exact, far cheaper per eval — golden-gated against the
    # full-graph path). Set general.oracle_local=false to use the original full-graph compute.
    use_local = bool(_cfg_get(cfg, "oracle_local", True))
    if use_local:
        engine = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=radius)
        node_oracle = lambda n: engine.compute_node_oracle(
            n, budget, topk=topk, candidate_strategy="tiered",
            brute_cutoff=brute_cutoff, beam_width=beam_width)
    else:
        global_cache = _global_cache(retriever, data)
        node_oracle = lambda n: compute_node_oracle(
            data, retriever, n, budget, topk, global_cache, radius, candidate_strategy="brute")
    for i, node in enumerate(tqdm(todo, desc=f"Oracle {cfg.dataset.name}/{cfg.retriever.name}")):
        oracle[node] = node_oracle(node)
        if (i + 1) % flush_every == 0:
            atomic_pickle_dump(oracle, file_path)
    atomic_pickle_dump(oracle, file_path)
    logger.info(f"Saved oracle for {len(oracle)} target nodes to {file_path}")


if __name__ == "__main__":
    main()
