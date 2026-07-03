import torch
from src.oracle.local_encode import LocalOracleEngine
from build_oracle import _incident_edges, _remove_edges_from_data


def test_local_embeddings_match_full_encode(cora_setup):
    cfg, data, retriever = cora_setup
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=cfg.dataset.query_radius)
    # pick a non-trivial high-degree node
    deg = torch.zeros(data.x.shape[0]).index_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
    node = int(deg.argmax())
    ctx = eng.prepare(node)
    edges = _incident_edges(data, node)[:3]                    # remove a few incident edges
    # full-graph attacked embedding (ground truth)
    full = retriever.encode(_remove_edges_from_data(data, edges).x,
                            _remove_edges_from_data(data, edges).edge_index)
    # localized affected embeddings
    loc = eng._local_affected_embeddings(ctx, edges)           # [n_affected, d], rows aligned to ctx.affected
    assert torch.allclose(full[ctx.affected], loc, atol=1e-5, rtol=0)


from build_oracle import _node_query, _position, _global_cache


def test_local_position_matches_full(cora_setup):
    cfg, data, retriever = cora_setup
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=cfg.dataset.query_radius)
    gcache = _global_cache(retriever, data)
    deg = torch.zeros(data.x.shape[0]).index_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
    order = deg.argsort(descending=True).tolist()
    # high-degree + mid-degree + node 259 (a known exact-cosine-tie case) to exercise tie-breaking
    nodes = order[:4] + order[len(order) // 2:len(order) // 2 + 3] + [259, 66, 213]
    for node in nodes:
        ctx = eng.prepare(node)
        q, init_full = _node_query(retriever, data, node, gcache, cfg.dataset.query_radius)
        assert ctx.init == init_full, (node, "init")                  # clean-graph rank matches (incl. ties)
        cands = _incident_edges(data, node)
        for combo in [cands[:1], cands[:2], cands[:3]]:
            full = _position(retriever, _remove_edges_from_data(data, combo), node, q)
            assert eng.position(ctx, combo) == full, (node, "pos")    # exact integer rank


from build_oracle import compute_node_oracle as full_compute, _global_cache as _gc


def test_compute_node_oracle_matches_full(cora_setup):
    cfg, data, retriever = cora_setup
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=cfg.dataset.query_radius)
    gcache = _gc(retriever, data)
    deg = torch.zeros(data.x.shape[0]).index_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
    for node in deg.argsort(descending=True)[:4].tolist():
        full = full_compute(data, retriever, int(node), budget=5, topk=15, global_cache=gcache,
                            radius=cfg.dataset.query_radius, candidate_strategy="topk")
        loc = eng.compute_node_oracle(int(node), budget=5, topk=15, candidate_strategy="topk")
        assert loc["init_position"] == full["init_position"]
        for b in range(1, 6):
            assert loc["optimal"][b]["position_change"] == full["optimal"][b]["position_change"]
            assert set(map(tuple, loc["optimal"][b]["edges"])) == set(map(tuple, full["optimal"][b]["edges"]))


def test_positions_batched_matches_sequential(cora_setup):
    cfg, data, retriever = cora_setup
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=cfg.dataset.query_radius)
    deg = torch.zeros(data.x.shape[0]).index_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
    node = int(deg.argmax())
    ctx = eng.prepare(node)
    cands = _incident_edges(data, node)
    combos = [cands[:1], cands[:2], cands[:3], cands[1:3]]
    seq = [eng.position(ctx, c) for c in combos]
    bat = eng.positions_batched(ctx, combos)
    assert bat == seq
