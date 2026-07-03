import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from src.data.splits import get_attack_split


def _rand_graph(n=300, avg_deg=6, seed=0):
    """Deterministic random undirected graph with varied degrees."""
    g = torch.Generator().manual_seed(seed)
    m = n * avg_deg // 2
    src = torch.randint(0, n, (m,), generator=g)
    dst = torch.randint(0, n, (m,), generator=g)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    ei = torch.unique(ei, dim=1)            # dedup, symmetric
    x = torch.randn(n, 8, generator=g)
    return Data(x=x, edge_index=ei)


def _eligible(data):
    n = data.x.shape[0]
    deg = degree(data.edge_index[0], num_nodes=n) + degree(data.edge_index[1], num_nodes=n)
    return torch.where(deg > deg.mean())[0].tolist(), deg


def test_determinism_same_seed():
    data = _rand_graph()
    assert get_attack_split(data, seed=42) == get_attack_split(data, seed=42)


def test_different_seed_changes_split():
    data = _rand_graph()
    assert get_attack_split(data, seed=42) != get_attack_split(data, seed=7)


def test_train_test_disjoint():
    data = _rand_graph()
    train, test = get_attack_split(data, seed=42)
    assert set(train).isdisjoint(set(test))


def test_all_targets_above_mean_degree():
    data = _rand_graph()
    elig, _ = _eligible(data)
    elig = set(elig)
    train, test = get_attack_split(data, seed=42)
    assert set(train) <= elig and set(test) <= elig


def test_sizes_match_ratios():
    data = _rand_graph()
    elig, _ = _eligible(data)
    E = len(elig)
    train, test = get_attack_split(data, seed=42, test_ratio=0.10, train_ratio=0.40)
    assert len(test) == max(1, int(E * 0.10))
    assert len(train) == max(1, int(E * 0.40))


import os
import json
import pytest
from src.data.splits import load_or_create_attack_split


def test_manifest_create_then_verify(tmp_path):
    data = _rand_graph()
    cache = str(tmp_path)
    tr1, te1 = load_or_create_attack_split(data, "toy", cache, seed=42)
    path = os.path.join(cache, "splits", "toy_seed42.json")
    assert os.path.exists(path)                              # manifest written on first call
    man = json.load(open(path))
    assert man["train_ids"] == tr1 and man["test_ids"] == te1
    assert man["seed"] == 42 and man["eligible_count"] > 0
    tr2, te2 = load_or_create_attack_split(data, "toy", cache, seed=42)
    assert (tr1, te1) == (tr2, te2)                          # second call verifies & matches


def test_manifest_drift_raises(tmp_path):
    data = _rand_graph()
    cache = str(tmp_path)
    load_or_create_attack_split(data, "toy", cache, seed=42)
    path = os.path.join(cache, "splits", "toy_seed42.json")
    man = json.load(open(path))
    man["test_ids"] = man["test_ids"][:-1] + [999999]        # simulate silent drift
    json.dump(man, open(path, "w"))
    with pytest.raises(RuntimeError):
        load_or_create_attack_split(data, "toy", cache, seed=42)
