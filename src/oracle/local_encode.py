"""Localized, EXACT oracle re-encoding engine.

Replaces each full-graph re-encode in the exhaustive oracle with a receptive-field-local
recompute that is bit-for-bit identical for the nodes whose embeddings can change. Validated
against the full-graph oracle by tests/test_oracle_golden.py.
"""
import itertools
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.oracle.query import neighborhood_centroid_query

# Per-node cost gate: if exhaustive brute over all incident edges would enumerate more than this many
# top-size combinations (C(deg, budget)), fall to BEAM even when deg <= brute_cutoff. Bounds a single
# node's wall-clock (a deg-50 budget-5 node is C(50,5)=2.1M combos ~ many hours, risking shard timeout)
# and is near-LOSSLESS: synergy ∝ 1/degree, so at the high degrees this triggers, beam ≈ the optimum.
# The per-graph brute_cutoff is the intended ceiling; this only refines it where brute is infeasible.
MAX_BRUTE_COMBOS = 800_000

# Beam expands over the top-scored incident edges only, to bound cost on very-high-degree nodes
# (deg≫cutoff) where exhaustive-over-all-edges is intractable. 64 ≫ the leak regime (deg 16-40),
# and synergy ∝ 1/degree means edges ranked beyond this carry negligible marginal/joint effect at
# high degree — so this never narrows the validated mid-degree beam (there deg ≤ 64 = all edges).
BEAM_CAND_CAP = 64


def k_hop_nodes(edge_index, node, k, num_nodes):
    """Sorted-ascending LongTensor of all nodes within `k` hops of `node` (inclusive of self)."""
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


def induced_subgraph_preserving_order(edge_index, nodes, num_nodes):
    """Edges with BOTH endpoints in `nodes`, kept in ORIGINAL column order, relabeled to
    contiguous local ids following `nodes`' order.

    Order preservation is what makes the localized re-encode bit-identical to the full-graph
    encode: a boolean column mask never reorders, and node relabeling is an elementwise map, so
    each node's incident-edge accumulation order in GCN message passing is unchanged.

    Returns (sub_edge_index_relabeled, node_map) with node_map[orig]=local id (-1 if absent).
    """
    device = edge_index.device
    in_sub = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    in_sub[nodes] = True
    row, col = edge_index
    emask = in_sub[row] & in_sub[col]                  # preserves column order
    sub = edge_index[:, emask]
    node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    node_map[nodes] = torch.arange(nodes.numel(), device=device)
    return node_map[sub], node_map


class LocalOracleEngine:
    """Exact, receptive-field-localized oracle compute. Public output matches the full-graph
    oracle (build_oracle.compute_node_oracle) bit-for-bit; verified by the golden test."""

    def __init__(self, retriever, data, num_layers=2, *, radius):
        self.r = retriever
        self.data = data
        self.L = int(num_layers)
        self.N = int(data.x.shape[0])
        self.radius = radius                                                  # centroid query neighborhood
        self.clean_emb = retriever.encode(data.x, data.edge_index).detach()   # [N, d]

    def prepare(self, node):
        """Per-node constants: affected set (<=L+1 hops), halo subgraph (<=2L+2 hops), the
        constant isolated query q, the clean baseline sims, and the clean-graph init rank.

        Halo is 2L+2 (NOT 2L+1): each GCNConv recomputes node degree from the edges it is given,
        so every node feeding an affected node's L-hop computation must have its FULL degree, which
        requires one hop beyond the affected nodes' L-hop receptive field. Empirically bit-identical
        on Cora+EPAGCL at 2L+2=6; inexact at 2L+1=5."""
        ei = self.data.edge_index
        affected = k_hop_nodes(ei, node, self.L + 1, self.N)
        halo = torch.sort(k_hop_nodes(ei, node, 2 * self.L + 2, self.N))[0]
        sub_ei, node_map = induced_subgraph_preserving_order(ei, halo, self.N)
        halo_x = self.data.x[halo]
        affected_local = node_map[affected]
        # constant FROZEN neighborhood-centroid query, from the surrogate's CLEAN embeddings
        # (does not move under attack). See src/oracle/query.py.
        q = neighborhood_centroid_query(ei, node, self.radius, self.clean_emb, self.N).detach()
        baseline_sims = F.cosine_similarity(q, self.clean_emb, dim=-1).squeeze(0)   # [N]
        init = self._rank(baseline_sims, node)
        return SimpleNamespace(node=node, affected=affected, affected_local=affected_local,
                               sub_ei=sub_ei, node_map=node_map, halo_x=halo_x,
                               q=q, baseline_sims=baseline_sims, init=init)

    def _remove_local(self, sub_ei, node_map, removed_edges):
        """Drop both directed entries of each removed undirected edge from the (local) subgraph,
        preserving column order. `removed_edges` are (orig_src, orig_dst) tuples."""
        if not removed_edges:
            return sub_ei
        pairs = []
        for a, b in removed_edges:
            la, lb = int(node_map[a]), int(node_map[b])
            pairs += [(la, lb), (lb, la)]
        rem = torch.tensor(sorted(pairs), dtype=torch.long, device=sub_ei.device)
        match = (sub_ei.t()[:, None, :] == rem[None, :, :]).all(-1).any(-1)
        return sub_ei[:, ~match]

    def _local_affected_embeddings(self, ctx, removed_edges):
        """Embeddings of the affected nodes (rows aligned to ctx.affected) after removing
        `removed_edges`, computed on the halo subgraph — exact for the affected (<=L+1 hop) set."""
        sub_ei = self._remove_local(ctx.sub_ei, ctx.node_map, removed_edges)
        sub_emb = self.r.encode(ctx.halo_x, sub_ei)
        return sub_emb[ctx.affected_local]

    def _rank(self, sims, node):
        """Retrieval rank of `node` = #docs strictly more similar, ties broken by ascending id —
        matches retrieve()'s argsort(descending=True) stable-ascending tie order exactly."""
        s = sims[node]
        idx = torch.arange(sims.shape[0], device=sims.device)
        return int(((sims > s) | ((sims == s) & (idx < node))).sum().item())

    def position(self, ctx, removed_edges):
        """Exact retrieval rank of ctx.node after removing `removed_edges`: only the affected nodes'
        sims are recomputed over the precomputed clean baseline; ties broken to match the oracle."""
        aff = self._local_affected_embeddings(ctx, removed_edges)
        sims = ctx.baseline_sims.clone()
        sims[ctx.affected] = F.cosine_similarity(ctx.q, aff, dim=-1)
        return self._rank(sims, ctx.node)

    def positions_batched(self, ctx, list_of_removed):
        """Exact ranks for many edge-removal sets in ONE batched forward (block-diagonal copies
        of the halo). Identical to [position(ctx, r) for r in list_of_removed]."""
        n = ctx.halo_x.shape[0]
        xs, eis = [], []
        for i, rem in enumerate(list_of_removed):
            eis.append(self._remove_local(ctx.sub_ei, ctx.node_map, rem) + i * n)
            xs.append(ctx.halo_x)
        bemb = self.r.encode(torch.cat(xs, 0), torch.cat(eis, 1))
        out = []
        for i in range(len(list_of_removed)):
            aff = bemb[i * n:(i + 1) * n][ctx.affected_local]
            sims = ctx.baseline_sims.clone()
            sims[ctx.affected] = F.cosine_similarity(ctx.q, aff, dim=-1)
            out.append(self._rank(sims, ctx.node))   # tie-aware, matches position()
        return out

    def compute_node_oracle(self, node, budget, *, topk=15, candidate_strategy="tiered",
                            brute_cutoff=0, beam_width=100):
        """Best incident-edge set to remove per budget for one target node.

        Semantics: optimal[b] = best demotion using 1..b edges (a budget CAP -> position_change is
        MONOTONE non-decreasing in b). candidate_strategy:
          "topk"   -> exhaustive over the top-`topk` single-edge-ranked incident edges (the path the
                      golden gate compares vs the full-graph reference; == v1 candidate pool).
          "tiered" -> deg <= brute_cutoff: exhaustive over ALL incident edges (true optimum);
                      else: beam(width) over all edges  UNION  top-`topk` exhaustive (hard dominance
                      over the top-K oracle). Verifier is positions_batched (bit-exact, golden-tested).
        """
        from build_oracle import _incident_edges
        ctx = self.prepare(node)
        candidates = _incident_edges(self.data, node)
        deg = len(candidates)
        cap = max(1, min(256, 200000 // max(1, ctx.halo_x.shape[0])))   # ~bounded batch memory
        if deg == 0:
            return {"init_position": ctx.init,
                    "optimal": {b: {"edges": [], "position_change": 0, "final_position": ctx.init}
                                for b in range(1, budget + 1)}}

        # single-edge demotion ranking over ALL incident edges (cheap; seeds top-K + beam)
        single = []
        for s in range(0, deg, cap):
            single += self.positions_batched(ctx, [[e] for e in candidates[s:s + cap]])
        scored = sorted(((single[i] - ctx.init, candidates[i]) for i in range(deg)), key=lambda x: -x[0])
        topk_edges = [e for _, e in scored[:topk]]

        def best_over(combo_iter):
            """Stream edge-sets through positions_batched in cap-sized chunks; return the
            (best_position, best_set). Never materializes the full combo list (C(deg,5) can be
            millions of tuples)."""
            best_pos, best_set, chunk = -1, None, []

            def _flush():
                nonlocal best_pos, best_set
                if not chunk:
                    return
                for c, pos in zip(chunk, self.positions_batched(ctx, chunk)):
                    if pos > best_pos:
                        best_pos, best_set = pos, c
                chunk.clear()

            for combo in combo_iter:
                chunk.append(list(combo))
                if len(chunk) >= cap:
                    _flush()
            _flush()
            return best_pos, best_set

        # brute over ALL edges only when within the per-graph cutoff AND the combo count is tractable
        # (else beam — see MAX_BRUTE_COMBOS). topk mode never brutes/beams.
        use_brute = (candidate_strategy == "tiered" and deg <= brute_cutoff
                     and math.comb(deg, min(budget, deg)) <= MAX_BRUTE_COMBOS)
        beam_out = None
        if candidate_strategy == "tiered" and not use_brute:
            beam_out = self._beam(ctx, scored, budget, beam_width, cap)

        # exhaustive pool: ALL edges when bruting, else the top-K (unioned with beam below)
        pool = candidates if use_brute else topk_edges

        optimal, run_pos, run_set = {}, -1, None
        for b in range(1, budget + 1):
            if b <= len(pool):                                  # exact-size-b over the pool
                bp, bs = best_over(itertools.combinations(pool, b))
                if bp > run_pos:
                    run_pos, run_set = bp, bs
            if beam_out is not None and beam_out[b][0] > run_pos:   # union beam (<=b)
                run_pos, run_set = beam_out[b]
            optimal[b] = {"edges": [tuple(map(int, e)) for e in run_set],
                          "position_change": run_pos - ctx.init, "final_position": run_pos}
        return {"init_position": ctx.init, "optimal": optimal}

    def _beam(self, ctx, scored, budget, beam_width, cap):
        """Beam over subset SIZE: keep top-W partial sets, expand each by one incident edge over ALL
        candidates (faithful to scripts/oracle_beam.py — expansion is NOT restricted to the seeds),
        exact-verify via positions_batched. Returns {b: (best_pos, best_set)} where budget b holds the
        best set of size 1..b found (monotone). Expansion is over the top-BEAM_CAND_CAP scored edges to
        bound cost on very-high-degree nodes (= all edges in the validated deg<=cutoff regime)."""
        expand_pool = [e for _, e in scored[:BEAM_CAND_CAP]]
        pos_cache = {}

        def _eval(setlists):
            todo = [s for s in setlists if frozenset(s) not in pos_cache]
            for i in range(0, len(todo), cap):
                chunk = todo[i:i + cap]
                for s, p in zip(chunk, self.positions_batched(ctx, chunk)):
                    pos_cache[frozenset(s)] = p

        out, beam, run_pos, run_set = {}, [[]], -1, None
        reach = min(budget, len(expand_pool))
        for b in range(1, reach + 1):
            seen, expanded = set(), []
            for S in beam:
                Sset = set(S)
                for e in expand_pool:
                    if e not in Sset:
                        key = frozenset(Sset | {e})
                        if key not in seen:
                            seen.add(key)
                            expanded.append(sorted(Sset | {e}))
            _eval(expanded)
            expanded.sort(key=lambda s: -pos_cache[frozenset(s)])
            beam = expanded[:beam_width]
            if beam:
                lvl_pos = pos_cache[frozenset(beam[0])]
                if lvl_pos > run_pos:
                    run_pos, run_set = lvl_pos, list(beam[0])
            out[b] = (run_pos, run_set)
        for b in range(reach + 1, budget + 1):                  # budgets beyond reachable size
            out[b] = (run_pos, run_set)
        return out
