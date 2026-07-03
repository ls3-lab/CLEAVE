#!/usr/bin/env python
"""Format baseline_summary.csv into the wide per-retriever-block table (project template):
three side-by-side blocks EPAGCL | GCA | GRACE, each `Dataset | Method | Budget 1..5`, with one
row per (dataset, method). Cell = mean rank-demotion (avg_position_change; positive = demoted).
Empty cell = not yet computed. (A proposed method like `cleave` would simply be one more Method row.)

Usage: python results/build_table.py
"""
import argparse, csv
import pandas as pd

RETRIEVERS = [("epagcl", "EPAGCL"), ("gca", "GCA"), ("grace", "GRACE")]
DATASETS = ["Cora", "CiteSeer", "PubMed", "DBLP", "CoraFull",
            "Coauthor-CS", "Amazon-Photo", "Amazon-Computers", "Coauthor-Physics"]
# (attacker-name-in-data, display-name) in the template's method order
METHODS = [("oracle", "oracle"), ("degree", "degree"), ("pagerank", "pagerank"),
           ("infattack_kl", "infattack_kl"), ("infattack_ul", "infattack_ul"), ("infmax", "infmax"),
           ("rlp", "rlp"), ("random", "random"), ("viking", "viking"), ("graphmi", "graphmi"),
           ("graphzoom", "graphzoom"),
           # Tier-1 gradient attackers: query-drift (the metric's query) vs self-drift (own clean emb, query-free)
           ("agpoa_querydrift", "agpoa (query-drift)"), ("agpoa_selfdrift", "agpoa (self-drift)"),
           ("nettack_querydrift", "nettack (query-drift)"), ("nettack_selfdrift", "nettack (self-drift)"),
           ("nag_r_querydrift", "nag_r (query-drift)"), ("nag_r_selfdrift", "nag_r (self-drift)")]
BUDGETS = [1, 2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/baseline_scores/baseline_summary.csv")
    ap.add_argument("--out", default="results/baseline_scores/baseline_table.csv")
    args = ap.parse_args()

    s = pd.read_csv(args.summary)
    val = {(r.retriever, r.dataset, r.attacker, int(r.budget)): round(r.mean_position_change, 2)
           for r in s.itertuples()}

    def cell(rt, ds, atk, b):
        v = val.get((rt, ds, atk, b))
        return "" if v is None else v

    # 3-row header: retriever band / "Budget" band / column names, repeated per block
    h1, h2, h3 = [], [], []
    for _, RT in RETRIEVERS:
        h1 += ["", "", RT, "", "", "", ""]
        h2 += ["", "", "Budget", "", "", "", ""]
        h3 += ["Dataset", "Method"] + [str(b) for b in BUDGETS]
    rows = [h1, h2, h3]

    for ds in DATASETS:
        for atk, disp in METHODS:
            row = []
            for rt, _ in RETRIEVERS:
                row += [ds, disp] + [cell(rt, ds, atk, b) for b in BUDGETS]
            rows.append(row)
        rows.append([""] * len(h3))  # blank separator between datasets

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {args.out}: {len(DATASETS)} datasets x {len(METHODS)} methods x 3 retrievers x 5 budgets")


if __name__ == "__main__":
    main()
