#!/usr/bin/env python
"""Materialize the per-(dataset, retriever) frozen neighborhood-centroid QUERY vectors.

The rank-demotion metric scores each target against a frozen query = the per-dataset neighborhood
centroid of the target, built on the CLEAN graph from the VICTIM retriever's embeddings (the exact
quantity evaluate_retrieval computes on the fly, src/evaluations/retrieval.py). This script saves those
query vectors to cache/queries/ so researchers can download them alongside the oracle (cache/oracle) and
splits (cache/splits) instead of recomputing. Deterministic; regenerable from the shipped checkpoints.

One file per (dataset, retriever): cache/queries/<Dataset>_<retriever>_seed<seed>_query.pkl
  {dataset, retriever, radius, seed, node_ids: [int], queries: float32 [n_targets, d]}
Targets = train ∪ test from the attack split (same coverage as the oracle caches).

Run on Slurm (needs the victim checkpoint + a GPU): sbatch scripts/build_queries.sh
"""
import os
import pickle

import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from src.data.loader import get_dataset
from src.data.splits import load_or_create_attack_split
from src.retrievers import get_retriever
from src.oracle.query import neighborhood_centroid_query

DATASETS = ["cora", "citeseer", "pubmed", "dblp", "corafull",
            "coauthor_cs", "amazon_photo", "amazon_computers", "coauthor_physics"]
RETRIEVERS = ["epagcl", "gca", "grace"]


def main():
    os.makedirs("cache/queries", exist_ok=True)
    total_bytes = 0
    for ds in DATASETS:
        for rt in RETRIEVERS:
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize(version_base=None, config_path="conf"):
                cfg = compose(config_name="config", overrides=[f"dataset={ds}", f"retriever={rt}"],
                              return_hydra_config=True)
                HydraConfig.instance().set_config(cfg)
                device = cfg.general.device if torch.cuda.is_available() else "cpu"
                data = get_dataset(cfg.dataset).to(device)[0]
                train_ids, test_ids = load_or_create_attack_split(
                    data, cfg.dataset.name, cfg.general.cache_dir, seed=cfg.general.seed)
                targets = sorted(set(train_ids) | set(test_ids))
                retr = get_retriever(cfg.retriever, cfg, data, device, use_oracle=False)  # VICTIM = eval model
                with torch.no_grad():
                    emb = retr.encode(data.x.to(device), data.edge_index.to(device)).detach()
                n, radius = data.num_nodes, cfg.dataset.query_radius
                Q = np.stack([
                    neighborhood_centroid_query(data.edge_index, t, radius, emb, n).squeeze(0).cpu().numpy()
                    for t in targets
                ]).astype(np.float32)
                out = f"cache/queries/{cfg.dataset.name}_{rt}_seed{cfg.general.seed}_query.pkl"
                with open(out, "wb") as f:
                    pickle.dump({"dataset": cfg.dataset.name, "retriever": rt, "radius": radius,
                                 "seed": int(cfg.general.seed), "node_ids": targets, "queries": Q}, f)
                sz = os.path.getsize(out); total_bytes += sz
                print(f"  {cfg.dataset.name}/{rt}: queries {Q.shape} radius={radius} -> {out} ({sz/1e6:.2f} MB)")
    print(f"DONE: cache/queries total {total_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
