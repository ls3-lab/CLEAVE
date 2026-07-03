# Baseline attack scores

Rank-demotion attack results for every baseline against the frozen GNN retrievers, on the
`seed=42` attack **test** split (`cache/splits/*_seed42.json`), budgets 1–5. These accompany the
oracle ceiling (`cache/oracle/`) and the splits so the baselines can be compared head-to-head.

## Files

- **`baseline_scores/baseline_summary.csv`** — one row per `(dataset, retriever, attacker, budget)`:
  - `mean_position_change` — mean rank demotion over the test targets. **Positive = demoted = a more
    successful attack** (`final_rank − init_rank`).
  - `success_rate` — fraction of targets with a positive demotion.
  - `n_nodes` — number of test targets aggregated.
- **`baseline_scores/coverage.csv`** — how many of the 17 attackers are present per `(retriever,dataset)`
  (`complete = all 17`). Use this to see what is finalized.
- **`build_summary.py`** — regenerates the summary from the raw per-row CSVs.

The **full per-row results** (every `node × budget × attacker`) and per-attacker timing are committed
under `per_node/` and `timing/` (27 files each) — see [Per-node results + timing](#per-node-results--timing-full-detail) below.

## Threat model / how to read

- Black-box transfer: attackers optimize on the **surrogate** (`best_ac_gcn.pth`); scored on the
  **victim** (`best_ac_gcn_attack.pth`). `oracle` is the brute-force ceiling (built on the surrogate).
- `graphmi` is the one **white-box** baseline (distills the victim) and is reported separately.
- `infattack_kl` (ground-truth labels) and `infattack_ul` (surrogate-predicted labels) are the two
  faithful InfAttack label settings.

## Known baseline behaviors (expected, not bugs)

- **InfAttack (`infattack_kl`/`infattack_ul`) goes negative on Amazon-Photo / Amazon-Computers.**
  This is faithful behavior, not a defect. InfAttack optimizes a *classification-margin* influence
  objective (Wang et al., WSDM 2024), which is **anti-aligned with this rank-demotion metric on dense,
  strongly-homophilous graphs**: it agrees with the oracle on sparse graphs (edge Jaccard ≈0.62, per-node
  corr ≈+0.83 on Cora) but picks near-disjoint, anti-correlated edges on the dense Amazon graphs (Jaccard
  ≈0.07, corr ≈−0.6) — even though demotion *is* achievable there (the oracle reaches +97/+214). It's a
  genuine objective-vs-metric transfer-gap result, amplified by the volatile/anisotropic embeddings on
  those datasets. Reported as-is.
- **`degree`/`pagerank` now remove HIGH-degree / HIGH-PageRank neighbors first.** The original low-importance
  variants were counterproductive (they promoted targets); the corrected high-importance versions are the
  ones reported here.

## Status: COMPLETE

All **17 attackers** across all **27** (retriever, dataset) combinations — `coverage.csv` is 27/27.
The full per-node results and timing are committed under `per_node/` and `timing/` (below).

## Per-node results + timing (full detail)

- `per_node/results_<Dataset>_<retriever>_5.csv` — per-(node, budget, attacker) results for all 17
  attackers (columns: `node_id, budget, attacker, init_position, final_positions, position_change,
  edges_removed, avg_position_change, ...`). 27 files, one per (dataset, retriever). These are the raw
  rows behind `baseline_scores/baseline_summary.csv` + `baseline_table.csv`.
- `timing/results_<Dataset>_<retriever>_5_timing.csv` — per-attacker wall-clock (`time_in_inference`,
  `retriever_surrogate_time`, `total_time_with_retriever_surrogate`).
