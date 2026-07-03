import torch
from src.oracle.local_encode import k_hop_nodes


def _line_graph(n=6):
    # 0-1-2-3-4-5 (undirected)
    pairs = [(i, i + 1) for i in range(n - 1)]
    ei = []
    for a, b in pairs:
        ei += [(a, b), (b, a)]
    return torch.tensor(ei, dtype=torch.long).t().contiguous()


def test_k_hop_includes_self_at_zero():
    ei = _line_graph()
    assert k_hop_nodes(ei, 2, 0, 6).tolist() == [2]


def test_k_hop_one_hop():
    ei = _line_graph()
    assert k_hop_nodes(ei, 2, 1, 6).tolist() == [1, 2, 3]


def test_k_hop_two_hops():
    ei = _line_graph()
    assert k_hop_nodes(ei, 2, 2, 6).tolist() == [0, 1, 2, 3, 4]


def test_k_hop_saturates():
    ei = _line_graph()
    assert k_hop_nodes(ei, 2, 10, 6).tolist() == [0, 1, 2, 3, 4, 5]


from src.oracle.local_encode import induced_subgraph_preserving_order


def test_induced_subgraph_preserves_order_and_relabels():
    ei = _line_graph(6)                       # cols: (0,1)(1,0)(1,2)(2,1)(2,3)(3,2)(3,4)(4,3)(4,5)(5,4)
    nodes = torch.tensor([1, 2, 3])
    sub, node_map = induced_subgraph_preserving_order(ei, nodes, 6)
    # only edges fully inside {1,2,3}, in ORIGINAL column order, relabeled 1->0,2->1,3->2
    assert sub.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert node_map[1].item() == 0 and node_map[2].item() == 1 and node_map[3].item() == 2
    assert node_map[0].item() == -1 and node_map[5].item() == -1
