"""Unit tests for the neighborhood-centroid query builder (CPU, toy graphs, no model)."""
import pytest
import torch

from src.oracle.query import neighborhood_centroid_query


def _ei(pairs):
    """Undirected edge_index from a list of {a,b} pairs (both directions)."""
    cols = []
    for a, b in pairs:
        cols += [(a, b), (b, a)]
    return torch.tensor(cols, dtype=torch.long).t().contiguous()


# graph: 0-1, 0-2, 2-3  ->  target 0: 1-hop {1,2}; (1∪2)-hop {1,2,3}
GRAPH = [(0, 1), (0, 2), (2, 3)]


def test_1hop_mean():
    emb = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [5., 5.]])
    q = neighborhood_centroid_query(_ei(GRAPH), 0, "1hop", emb, 4)
    assert q.shape == (1, 2)
    assert torch.allclose(q[0], torch.tensor([0.5, 0.5]))          # mean of nodes 1,2


def test_2hop_mean():
    emb = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [5., 5.]])
    q = neighborhood_centroid_query(_ei(GRAPH), 0, "2hop", emb, 4)
    assert torch.allclose(q[0], torch.tensor([2.0, 2.0]))          # mean of nodes 1,2,3 (target excluded)


def test_blend():
    emb = torch.tensor([[2., 2.], [1., 0.], [0., 1.], [5., 5.]])
    q = neighborhood_centroid_query(_ei(GRAPH), 0, "blend", emb, 4)
    c1 = torch.tensor([0.5, 0.5])
    assert torch.allclose(q[0], (torch.tensor([2., 2.]) + c1) / 2)


@pytest.mark.parametrize("radius", ["1hop", "2hop", "blend"])
def test_isolated_node_fallback(radius):
    emb = torch.randn(4, 3)
    ei = _ei([(1, 2)])                                             # node 0 isolated
    q = neighborhood_centroid_query(ei, 0, radius, emb, 4)
    assert torch.allclose(q[0], emb[0])                           # falls back to own embedding


def test_zero_norm_fallback():
    emb = torch.tensor([[7., 1.], [1., 0.], [-1., 0.]])           # neighbors 1,2 cancel -> centroid ~0
    q = neighborhood_centroid_query(_ei([(0, 1), (0, 2)]), 0, "1hop", emb, 3)
    assert torch.allclose(q[0], emb[0])                           # zero-norm guard -> own embedding


def test_unknown_radius_raises():
    emb = torch.randn(2, 4)
    with pytest.raises(ValueError):
        neighborhood_centroid_query(_ei([(0, 1)]), 0, "3hop", emb, 2)


def test_full_embedding_assertion():
    with pytest.raises(AssertionError):
        neighborhood_centroid_query(_ei([(0, 1)]), 0, "1hop", torch.randn(5, 3), 2)  # rows != num_nodes
