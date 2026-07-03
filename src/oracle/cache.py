"""Single source of truth for the oracle cache filename.

Imported by BOTH the writer (build_oracle.py) and the reader (src/attackers/oracle.py) so the
name can never drift. The `v2` tag + (brute_cutoff, beam_width) tokens mark the degree-tiered
oracle (centroid query + brute/beam search), so a stale `_K{topk}_` v1 cache is never loaded
silently — the method now defines the file, not the old top-K knob.
"""


def oracle_cache_filename(retriever_upper, budget, brute_cutoff, beam_width, dataset_name, graph_model):
    return (f"{retriever_upper}_Oracle_v2_Budget{budget}_cut{brute_cutoff}_W{beam_width}_"
            f"{dataset_name}_{graph_model}.pkl")
