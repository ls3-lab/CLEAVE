import os
import torch
import pytest
from hydra import initialize_config_dir, compose
from hydra.core.hydra_config import HydraConfig
from src.data.loader import get_dataset
from src.retrievers import get_retriever
from src.data.splits import get_attack_split
from build_oracle import compute_node_oracle as full_compute, _global_cache
from src.oracle.local_encode import LocalOracleEngine

# Opt-in: the gate runs the SLOW full-graph oracle for comparison (~1 min/node on CPU), so it is
# skipped unless RUN_GOLDEN=1. Uses GPU if available (same-device full-vs-localized is still exact).
# budget=3 suffices: per-eval exactness is budget-independent; budget-5 was certified on Cora by the
# T5/T7 tests. This gate adds cross-dataset (CiteSeer) + cross-encoder (gca/grace) coverage.
GOLDEN = os.environ.get("RUN_GOLDEN") == "1"


def _golden(dataset, retriever_name, budget=3, n_sample=3, seed=42):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("conf")):
        cfg = compose(config_name="config",
                      overrides=[f"dataset={dataset}", f"retriever={retriever_name}"],
                      return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    data = get_dataset(cfg.dataset)[0].to(dev)
    retriever = get_retriever(cfg.retriever, cfg, data, dev, use_oracle=True)
    train_ids, test_ids = get_attack_split(data, seed=cfg.general.seed)
    nodes = sorted(set(train_ids) | set(test_ids))
    g = torch.Generator().manual_seed(seed)
    nodes = [nodes[i] for i in torch.randperm(len(nodes), generator=g)[:n_sample].tolist()]
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=cfg.dataset.query_radius)
    gcache = _global_cache(retriever, data)
    for node in nodes:
        # topk-vs-topk: the golden gate pins the localized EVALUATOR's exactness, not the tiered
        # search (brute != top-15 on deg>15 nodes -> would fail legitimately). Centroid query now.
        full = full_compute(data, retriever, node, budget=budget, topk=15, global_cache=gcache,
                            radius=cfg.dataset.query_radius, candidate_strategy="topk")
        loc = eng.compute_node_oracle(node, budget=budget, topk=15, candidate_strategy="topk")
        assert loc["init_position"] == full["init_position"], (dataset, retriever_name, node, "init")
        for b in range(1, budget + 1):
            assert loc["optimal"][b]["position_change"] == full["optimal"][b]["position_change"], (dataset, retriever_name, node, b, "Δ")
            assert set(map(tuple, loc["optimal"][b]["edges"])) == set(map(tuple, full["optimal"][b]["edges"])), (dataset, retriever_name, node, b, "edges")


@pytest.mark.skipif(not GOLDEN, reason="slow; set RUN_GOLDEN=1 to run the full-vs-localized golden gate")
@pytest.mark.parametrize("retriever_name", ["epagcl", "gca", "grace"])
@pytest.mark.parametrize("dataset", ["cora", "citeseer"])
def test_golden(dataset, retriever_name):
    _golden(dataset, retriever_name)
