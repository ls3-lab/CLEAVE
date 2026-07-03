#!/usr/bin/env python
"""Build the baseline scores summary from per-(retriever,dataset) attack result CSVs.

Reads the raw per-(node,budget,attacker) result CSVs (produced by retriever_attack.py) and aggregates
them into results/baseline_scores/baseline_summary.csv — the per-(dataset, retriever, attacker, budget)
mean rank-demotion (`avg_position_change`; POSITIVE = demoted = successful attack), success rate, and
node count. Also writes coverage.csv (how many of the 17 attackers are present per (retriever,dataset)).

The fresh C2 run is sharded into resource groups, each writing its own output_dir; pass those dirs (or
the merged output/) via --dirs. Re-run any time to refresh as more results land.

Usage:
  python results/build_summary.py --dirs output_light output_infmax output_grad output_graphmi
  python results/build_summary.py --dirs output            # after merge
"""
import argparse, glob, os
from collections import defaultdict
import pandas as pd

ALL_ATTACKERS = ["oracle", "degree", "pagerank", "random", "rlp", "viking", "graphzoom",
                 "infmax", "infattack_kl", "infattack_ul", "graphmi",
                 # Tier-1 gradient attackers: query-drift (aligned with the metric) + self-drift (query-free)
                 "agpoa_querydrift", "agpoa_selfdrift", "nettack_querydrift", "nettack_selfdrift",
                 "nag_r_querydrift", "nag_r_selfdrift"]

# Pre-rename CSVs labeled the Tier-1 query-drift attackers bare (agpoa/nettack/nag_r); relabel to the
# explicit *_querydrift name (identical attack — the runs predate the rename, no re-run needed).
NAME_REMAP = {"agpoa": "agpoa_querydrift", "nettack": "nettack_querydrift", "nag_r": "nag_r_querydrift"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["output"], help="output dir(s) holding attacks/results_*_5.csv")
    ap.add_argument("--out", default="results/baseline_scores")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # An attacker re-run into a dedicated dir supersedes its (stale) rows elsewhere. degree and pagerank
    # were re-run into output_degree / output_pagerank (high-importance-first fixes); drop the old
    # low-importance rows from output_light so only the corrected re-run results count.
    STALE = {"output_light": ["degree", "pagerank"]}

    combos = defaultdict(list)
    for d in args.dirs:
        for p in glob.glob(os.path.join(d, "attacks", "results_*_5.csv")):
            combos[os.path.basename(p)].append((os.path.basename(d.rstrip("/")), p))

    rows = []
    for base, dps in sorted(combos.items()):
        parts = []
        for dname, p in dps:
            df = pd.read_csv(p)
            df["attacker"] = df["attacker"].replace(NAME_REMAP)
            drop = STALE.get(dname, [])
            if drop:
                df = df[~df["attacker"].isin(drop)]
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        g = df.groupby(["dataset_name", "retriever_name", "attacker", "budget"])
        s = g["avg_position_change"].agg(mean_position_change="mean", n_nodes="count").reset_index()
        s["success_rate"] = g.apply(lambda x: (x["avg_position_change"] > 0).mean(), include_groups=False).values
        rows.append(s)
    if not rows:
        print("no result CSVs found"); return
    summary = (pd.concat(rows, ignore_index=True)
               .rename(columns={"dataset_name": "dataset", "retriever_name": "retriever"})
               [["dataset", "retriever", "attacker", "budget", "mean_position_change", "success_rate", "n_nodes"]])
    summary.to_csv(os.path.join(args.out, "baseline_summary.csv"), index=False)

    cov = summary.groupby(["dataset", "retriever"]).attacker.nunique().reset_index(name="n_attackers")
    cov["complete"] = cov.n_attackers == len(ALL_ATTACKERS)
    cov.to_csv(os.path.join(args.out, "coverage.csv"), index=False)
    print(f"baseline_summary.csv: {len(summary)} rows, attackers={sorted(summary.attacker.unique())}")
    print(f"coverage: {cov.complete.sum()}/{len(cov)} (retriever,dataset) complete (all {len(ALL_ATTACKERS)})")

if __name__ == "__main__":
    main()
