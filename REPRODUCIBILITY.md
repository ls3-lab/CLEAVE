# Reproducing the paper

This document lists the commands to reproduce the results in
*Contrastive Surrogate Attacks on Graph-Based Retrieval* (CLEAVE). All runs use
`seed = 42` and Hydra config composition. Retriever checkpoints are regenerated
from the seeded training scripts (they are not shipped); the oracle reference
cache, frozen queries, and target splits are included under `cache/`.

## 0. Environment

```bash
pip install -r requirements.txt
# torch-geometric needs compiled extensions (torch-scatter, torch-sparse, ...)
# matching your Torch/CUDA build. Install those per the PyG docs.
```

## 1. Train the victim retrievers

Attacks are evaluated against separately trained victim retrievers. Train the
three contrastive encoders per dataset (this produces
`cache/checkpoints/<retriever>/<Dataset>/best_ac_gcn.pth` and `..._attack.pth`).

```bash
# GCA
cd src/retrievers/GCA/ && python train.py --dataset Cora
# EPAGCL
python src/retrievers/EPAGCL/main.py --dataset Cora
# GRACE
python src/retrievers/GRACE/train.py --dataset Cora
```

Each encoder under `src/retrievers/` has its own training entry point; see the
respective folder for the exact dataset flags. Training is seeded, so the
retrained weights reproduce the checkpoints used to build the shipped frozen
queries and oracle reference.

## 2. (Optional) Rebuild the oracle reference

The optimal-removal reference (an analysis upper bound; CLEAVE never uses it) is
shipped under `cache/oracle/`. To rebuild it for a cell:

```bash
python build_oracle.py dataset=cora retriever=epagcl +general.force_rewrite=true
```

## 3. Main result: CLEAVE vs. twelve baselines

Run the full suite (CLEAVE plus the twelve baselines from `conf/config.yaml`)
for one cell, or sweep all 27 cells. Position change is written to
`output/<date>/<time>/results_*.csv`.

```bash
# single cell
python retriever_attack.py dataset=pubmed retriever=epagcl general.budget=5

# all 27 dataset x victim cells
bash scripts/run_full_suite.sh
```

## 4. CLEAVE only

```bash
# one cell
python retriever_attack.py --config-name config_cleave dataset=amazon_photo retriever=epagcl

# all datasets x victims
bash scripts/run_cleave.sh
```

The per-dataset surrogate configuration (contrastive edge-drop rates, epochs,
surrogate dimension `d`, degree exponent `beta`, and query radius `r`) selected
once on a held-out validation split is documented in the README ("Per-dataset
configuration") and set through `conf/attacker/cleave.yaml` overrides. The
scripts in `scripts/` apply these per-dataset overrides.

## 5. Sensitivity analyses

```bash
# surrogate embedding dimension d in {128, 256, 512}
bash scripts/ablation_surrogate_dim.sh

# contrastive augmentation (edge-drop rate)
bash scripts/ablation_edge_drop.sh
```

Each override is passed on the command line, for example:

```bash
python retriever_attack.py --config-name config_cleave dataset=pubmed retriever=epagcl \
  attackers.1.surrogate_hidden=512 attackers.1.surrogate_out=512
python retriever_attack.py --config-name config_cleave dataset=pubmed retriever=epagcl \
  attackers.1.drop1=0.0 attackers.1.drop2=0.0
```

## 6. Aggregating results

The published baseline score tables live under `results/`; regenerate the
summary and comparison tables with:

```bash
python results/build_summary.py
python results/build_table.py
```
