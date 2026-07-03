from build_oracle import shard_nodes


def test_shards_partition_without_overlap():
    nodes = list(range(100))
    shards = [shard_nodes(nodes, i, 4) for i in range(4)]
    flat = sorted(n for s in shards for n in s)
    assert flat == nodes                       # complete coverage
    seen = set()
    for s in shards:
        assert seen.isdisjoint(s)              # no overlap
        seen.update(s)


def test_single_shard_is_identity():
    nodes = list(range(7))
    assert shard_nodes(nodes, 0, 1) == nodes


def test_merge_shard_caches_unions_parts(tmp_path):
    import pickle
    from build_oracle import merge_shard_caches, atomic_pickle_dump
    canon = str(tmp_path / "ORACLE_x.pkl")
    atomic_pickle_dump({1: {"a": 1}, 2: {"a": 2}}, canon[:-4] + ".part0.pkl")
    atomic_pickle_dump({3: {"a": 3}}, canon[:-4] + ".part1.pkl")
    n_parts, n_nodes = merge_shard_caches(canon)
    assert n_parts == 2 and n_nodes == 3
    with open(canon, "rb") as f:
        merged = pickle.load(f)
    assert set(merged.keys()) == {1, 2, 3}
