"""Shared retrieval-query builder.

The rank-demotion metric needs a query embedding for the target node that is FROZEN — computed
once on the clean graph so it does NOT move when edges are removed under attack. The original
"isolated-target" query (the target's own embedding with all its incident edges dropped) was a
non-standard self-query; this module replaces it with a per-dataset NEIGHBORHOOD-CENTROID query
that is the locked design (see plan oracle-query-finalization / memory external-query-design).

Both the evaluator (src/evaluations/retrieval.py, victim embeddings) and the oracle builders
(build_oracle.py + src/oracle/local_encode.py, surrogate embeddings) import this one builder, so
the neighborhood definition is identical across them — only the model whose clean embeddings are
passed in (`clean_emb`) differs. The radius is per-dataset (conf/dataset/<name>.yaml: query_radius).

Formulas (mirror the validated scripts/k_check_slurm.py:query() and scripts/oracle_beam.py:qof()):
  1hop  -> mean of clean embeddings over the target's 1-hop neighbors
  2hop  -> mean over the (1-hop ∪ 2-hop) neighborhood, target excluded
  blend -> (target's own embedding + 1-hop centroid) / 2
Means are over node SETS, so order does not matter (commutative) -> bit-identical to the scripts.
"""
import torch


def _khop_nodes(edge_index, node, k, num_nodes):
    """Sorted-ascending LongTensor of all nodes within `k` undirected hops of `node` (incl. self).

    Mirrors src.oracle.local_encode.k_hop_nodes (kept local to avoid a circular import: local_encode
    imports this module). Relies on edge_index being undirected/symmetric (the framework invariant).
    """
    device = edge_index.device
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    visited[node] = True
    frontier = torch.tensor([node], device=device, dtype=torch.long)
    row, col = edge_index
    for _ in range(k):
        mask = torch.isin(row, frontier)
        nbrs = col[mask]
        nbrs = nbrs[~visited[nbrs]]
        if nbrs.numel() == 0:
            break
        visited[nbrs] = True
        frontier = torch.unique(nbrs)
    return torch.where(visited)[0]


def neighborhood_centroid_query(edge_index, target, radius, clean_emb, num_nodes):
    """Frozen centroid query for `target` from CLEAN embeddings `clean_emb` ([num_nodes, d]).

    Returns a [1, d] tensor. Falls back to the target's own embedding when the neighborhood is
    empty (isolated node) or the centroid degenerates to a ~zero / non-finite vector (cosine vs a
    zero vector is NaN, which _rank would silently treat as rank 0).
    """
    assert clean_emb.shape[0] == num_nodes, (
        f"clean_emb has {clean_emb.shape[0]} rows but num_nodes={num_nodes}; "
        "pass FULL-graph embeddings here, not the localized halo subset.")
    target = int(target)
    tgt_emb = clean_emb[target]

    if radius in ("1hop", "blend"):
        nbrs = _khop_nodes(edge_index, target, 1, num_nodes)
        nbrs = nbrs[nbrs != target]                       # 1-hop neighbors only
        if nbrs.numel() == 0:
            q = tgt_emb
        else:
            c1 = clean_emb[nbrs].mean(0)
            q = c1 if radius == "1hop" else (tgt_emb + c1) / 2
    elif radius == "2hop":
        nbrs = _khop_nodes(edge_index, target, 2, num_nodes)
        nbrs = nbrs[nbrs != target]                       # 1-hop ∪ 2-hop, target excluded
        q = clean_emb[nbrs].mean(0) if nbrs.numel() else tgt_emb
    else:
        raise ValueError(f"unknown query_radius {radius!r} (expected 1hop | 2hop | blend)")

    if not torch.isfinite(q).all() or torch.linalg.vector_norm(q) < 1e-8:
        q = tgt_emb
    return q.unsqueeze(0)
