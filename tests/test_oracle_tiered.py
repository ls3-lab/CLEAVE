"""Correctness tests for the degree-tiered oracle search (Cora + EPAGCL oracle model, CPU).

These pin the TIERED candidate generation independently of the golden gate (which only certifies
the localized evaluator via the top-K path). Kept light (mid/low degree, modest budgets) so they
run in the default suite; the localized evaluator's bit-exactness is what they lean on.
"""
import torch

from src.oracle.local_encode import LocalOracleEngine
from build_oracle import (_incident_edges, _node_query, _position, _remove_edges_from_data,
                          compute_node_oracle as full_compute, _global_cache)


def _deg(data):
    return torch.zeros(data.x.shape[0]).index_add_(
        0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))


def _nodes_in(data, lo, hi, k):
    deg = _deg(data)
    return [int(n) for n in range(data.x.shape[0]) if lo <= int(deg[n]) <= hi][:k]


def test_low_degree_brute_matches_full_exhaustive(cora_setup):
    """deg<=8: localized brute over ALL edges == full-graph brute (evaluator-exact on the brute path)."""
    cfg, data, retriever = cora_setup
    r = cfg.dataset.query_radius
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=r)
    gcache = _global_cache(retriever, data)
    nodes = _nodes_in(data, 3, 8, 5)
    assert nodes
    for node in nodes:
        loc = eng.compute_node_oracle(node, budget=5, topk=15, candidate_strategy="tiered",
                                      brute_cutoff=100, beam_width=100)          # deg<=8<100 -> brute
        full = full_compute(data, retriever, node, budget=5, topk=15, global_cache=gcache,
                            radius=r, candidate_strategy="brute")
        assert loc["init_position"] == full["init_position"], node
        for b in range(1, 6):
            assert loc["optimal"][b]["position_change"] == full["optimal"][b]["position_change"], (node, b)


def test_tiered_dominates_topk(cora_setup):
    """tiered (beam ∪ top-K) >= top-K alone, per budget — hard dominance over the old oracle."""
    cfg, data, retriever = cora_setup
    r = cfg.dataset.query_radius
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=r)
    nodes = _nodes_in(data, 12, 22, 4)
    assert nodes
    for node in nodes:
        tiered = eng.compute_node_oracle(node, budget=5, topk=15, candidate_strategy="tiered",
                                         brute_cutoff=10, beam_width=100)        # deg>10 -> beam∪topk
        topk = eng.compute_node_oracle(node, budget=5, topk=15, candidate_strategy="topk")
        for b in range(1, 6):
            assert tiered["optimal"][b]["position_change"] >= topk["optimal"][b]["position_change"], (node, b)


def test_beam_not_exceeds_brute(cora_setup):
    """beam best <= true brute optimum (beam never overshoots the optimum)."""
    cfg, data, retriever = cora_setup
    r = cfg.dataset.query_radius
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=r)
    nodes = _nodes_in(data, 12, 15, 3)
    assert nodes
    for node in nodes:
        beam = eng.compute_node_oracle(node, budget=4, topk=15, candidate_strategy="tiered",
                                       brute_cutoff=0, beam_width=100)           # deg>0 -> beam∪topk
        brute = eng.compute_node_oracle(node, budget=4, topk=15, candidate_strategy="tiered",
                                        brute_cutoff=1000, beam_width=100)        # deg<=1000 -> brute (opt)
        for b in range(1, 5):
            assert beam["optimal"][b]["position_change"] <= brute["optimal"][b]["position_change"], (node, b)


def test_chosen_sets_evaluator_exact(cora_setup):
    """The full-graph rank of each chosen optimal set matches the localized engine's reported rank."""
    cfg, data, retriever = cora_setup
    r = cfg.dataset.query_radius
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=r)
    gcache = _global_cache(retriever, data)
    nodes = _nodes_in(data, 12, 20, 3)
    assert nodes
    for node in nodes:
        q, _ = _node_query(retriever, data, node, gcache, r)
        rec = eng.compute_node_oracle(node, budget=5, topk=15, candidate_strategy="tiered",
                                      brute_cutoff=0, beam_width=100)
        for b in range(1, 6):
            edges = rec["optimal"][b]["edges"]
            full_pos = _position(retriever, _remove_edges_from_data(data, edges), node, q)
            assert full_pos == rec["optimal"][b]["final_position"], (node, b)


def test_monotone_in_budget(cora_setup):
    """position_change is non-decreasing in budget (the <=b cap semantics)."""
    cfg, data, retriever = cora_setup
    r = cfg.dataset.query_radius
    eng = LocalOracleEngine(retriever, data, num_layers=int(cfg.retriever.num_layers), radius=r)
    for node in _nodes_in(data, 12, 22, 4):
        rec = eng.compute_node_oracle(node, budget=5, topk=15, candidate_strategy="tiered",
                                      brute_cutoff=10, beam_width=100)
        pcs = [rec["optimal"][b]["position_change"] for b in range(1, 6)]
        assert all(pcs[i] <= pcs[i + 1] for i in range(len(pcs) - 1)), (node, pcs)
