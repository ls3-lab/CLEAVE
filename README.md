# Rank Demotion in Graph Retrieval via Edge Addition

**Sink any node in a GNN retriever's ranking by *adding* a handful of edges, without ever querying the victim.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](#installation)

Implementation of **Undercut**, a **query-free black-box** adversarial **rank-demotion** attack on
graph-based retrieval that works by **edge addition**, together with the evaluation framework: twelve
baseline attacks adapted to the add-edge setting and three unsupervised graph contrastive retrievers
(GRACE, GCA, EPAGCL).

## TL;DR

A frozen GNN retriever ranks nodes by embedding similarity. Undercut inserts up to
**b = 1 to 5 new undirected edges** incident to a **target node** and pushes that node **down** the
ranking. It never queries the victim, never sees labels, and never uses oracle information. It trains
its **own** self-supervised contrastive GCN surrogate from scratch on nothing but **public graph
structure and node features**, forms a compact **anchor shortlist** of candidate endpoints, and selects
the edge set with a **greedy add–recompute–rescore** search. Each candidate's effect is scored by a
**closed-form degree-weighted update of the surrogate embeddings** — no per-candidate forward pass — so
the attack is fast and scales to large graphs. The chosen insertions **transfer** unchanged to the
separately trained black-box victim.

> **Metric.** Position change `delta_t = rank_after(t) - rank_before(t)`.
> **Positive means the node moved down the ranking, i.e. a successful attack.**

## How Undercut works

1. **Train a surrogate encoder from scratch.** A **two-layer GCN** trained with a **GRACE-style NT-Xent**
   contrastive objective (edge-dropping and feature-masking augmentations) on public topology and node
   features only — no labels, no oracle, no victim access. The surrogate is trained once per dataset and
   reused across targets, budgets, and victims.

2. **Freeze the query.** For each target `t`, a **neighborhood-centroid query** `q_t` is computed once on
   the clean graph in the surrogate embedding space (mean of the `1-hop`, `2-hop`, or blended neighbor
   embeddings; radius fixed per dataset). The frozen `q_t` does not move under perturbation, so demotion
   is modeled as pushing the target *away* from its clean retrieval context.

3. **Form an anchor shortlist.** Candidate endpoints are ranked two ways and unioned: a surrogate-free
   **analytic low-pass score** (the closed-form `Â^k X` demotion criterion) and a **one-step surrogate
   gradient**. Restricting the search to this shortlist keeps selection cheap on large graphs.

4. **Greedy add–recompute–rescore.** Insert `b` edges one at a time; at each step pick the shortlist
   endpoint whose addition maximizes the exact set demotion, then recompute and rescore. The per-candidate
   marginal is evaluated analytically — adding edge `(t, j)` changes the target's 1-hop aggregate by a
   closed-form degree reweighting of the precomputed surrogate embeddings:

   ```
   z_t(S) ≈ sqrt(d_t / (d_t + |S|)) · Z_t  +  Σ_{j∈S} 1 / sqrt((d_t + |S|)(d_j + 1)) · Z_j
   loss   = 1 − cos(z_t(S), q_t)
   ```

The recorded insertions are applied to the graph and evaluated only on the **separately trained** victim
retriever, so transfer is a genuine empirical property rather than a result of querying the victim.

### Why this is genuinely query-free black-box

* The victim retriever is **never queried** during surrogate training or edge selection.
* No node labels and no target-specific oracle labels are used at attack time.
* The demotion target is the frozen `q_t`, not a victim response.
* The surrogate is trained only on public topology and node features.

## Rank-demotion setup

* **Retriever** (GRACE / GCA / EPAGCL): a frozen, pre-trained contrastive GCN encoder mapping each node to an embedding.
* **Retrieval and ranking**: the *query* is a frozen neighborhood-centroid built on the clean graph; the *corpus* is every node's embedding. Nodes are ranked by cosine similarity to the query. `position_change = rank_after - rank_before` (higher = more demoted). See `src/evaluations/retrieval.py`.
* **Attack**: an `Attacker` selects up to `budget` **new** undirected edges incident to the target and inserts them; the victim re-embeds the attacked graph and we measure the rank change.
* **Two victim models by design**: attacks are constructed against one model and evaluated against a **separately trained** victim, so results measure black-box transfer rather than white-box overfitting.

> **Graph convention:** all bundled datasets are **undirected**. An added edge is the undirected pair `{target, neighbor}`; `budget` counts undirected edges; insertion adds **both** directed entries, skips self-loops, and dedups against existing edges.

## Key results

At **budget 5**, Undercut produces the **largest rank demotion across the reported dataset × victim
settings**, outperforming the strongest baseline on every setting (by roughly **108% to 385%** at the
dataset level), with the advantage growing as the insertion budget increases. The analytic
degree-update scorer makes it **fast** — a few hundred milliseconds per target with a 1%-of-graph
shortlist — and **robust**: demotion is flat down to a 1% shortlist, and degrades gracefully under
random candidate-access restriction (a broad, redundant set of vulnerable edges rather than a few
specific ones).

## Baselines

Undercut is compared against **twelve baselines** spanning the main families of graph attacks, all
adapted to the add-edge retrieval setting (target-incident undirected edge *insertion*, same budget,
evaluated on the separately trained victim):

| Family | Baselines |
| --- | --- |
| Structural and influence | Random, Degree, PageRank, InfMax, Viking |
| Link-perturbation and black-box | RLP, GraphMI |
| Published and gradient-oriented add-edge | DICE, SPAC, PGA, AGPOA, NAG-R |

The gradient baselines use a proxy or surrogate for their gradients, kept separate from the held-out
victim, so the comparison is conservative with respect to Undercut's stricter access model.

> **Note (GraphMI).** GraphMI optimizes a dense `N×N` adjacency via projected gradient descent, so its
> memory scales quadratically with the number of nodes; on the largest-node graphs it exceeds GPU
> memory and is reported as **OOM** rather than run with a non-comparable sparse/reduced-precision variant.

## Repository structure

```
.
├── README.md                     # this file
├── requirements.txt              # torch, torch-geometric, hydra-core, ...
├── retriever_attack.py           # entry: run attackers on black-box retrievers, writes output_addedge/ CSVs
├── conf/                         # Hydra configs
│   ├── config_addedge.yaml       # generic add-edge entry (override attacker@attackers.1=<name>)
│   ├── config_addedge_undercut.yaml  # Undercut entry point
│   ├── dataset/                  # 9 datasets
│   ├── retriever/                # epagcl.yaml, gca.yaml, grace.yaml
│   └── attacker/                 # baseline configs plus undercut.yaml (Undercut)
├── src/
│   ├── attackers/
│   │   ├── base.py               # Attacker contract (add_edge mode)
│   │   ├── random.py, degree.py, pagerank.py, infmax.py, viking.py, rlp.py, graphmi.py, agpoa.py, nag_r.py
│   │   ├── __init__.py           # attacker registry
│   │   └── contrib/
│   │       ├── undercut.py        # Undercut, class UndercutAttack (proposed method)
│   │       ├── _analytic_utils.py   # analytic Â^k X propagation + shortlist score (self-contained)
│   │       └── published_addedge.py # DICE, SPAC, PGA add-edge baselines
│   ├── retrievers/               # EPAGCL / GCA / GRACE encoders plus registry
│   ├── data/                     # PyG loader, train/test target splits
│   ├── evaluations/retrieval.py  # centroid-query ranking, position_change metric
│   └── utils/
└── cache/
    ├── checkpoints/{epagcl,gca,grace}/<Dataset>/  # victim / reference weights (train via src/retrievers/; not bundled)
    ├── queries/                  # frozen centroid queries (seed 42)
    └── splits/                   # train/test target splits (seed 42)
```

## Installation

Requires **Python 3.8+**.

```bash
pip install -r requirements.txt
# torch-geometric needs compiled extensions (torch-scatter, torch-sparse, ...)
# matching your Torch/CUDA build. Install those per the PyG docs.
```

**Retriever checkpoints (not bundled).** Trained victim and surrogate weights are **not** shipped
with the repository. Train them from the deterministic, seeded scripts under `src/retrievers/`
(see [Training retrievers](#training-retrievers)); they are written to
`cache/checkpoints/<retriever>/<Dataset>/`. Undercut trains its own surrogate from scratch and needs
only the **victim** checkpoint to evaluate an attack.

## Quickstart

```bash
# 1) Run Undercut on PubMed against the GCA victim (budgets 1 to 5)
python retriever_attack.py --config-name config_addedge_undercut dataset=pubmed retriever=gca general.budget=5

# 2) Sweep Undercut across all three victim retrievers on Coauthor-CS
python retriever_attack.py --config-name config_addedge_undercut dataset=coauthor_cs --multirun retriever=epagcl,gca,grace

# 3) Run a baseline (e.g. PGA) on the generic add-edge config
python retriever_attack.py --config-name config_addedge dataset=pubmed retriever=gca general.budget=5 attacker@attackers.1=pga_add
```

Results land in `output_addedge/attacks/` as `results_add_*.csv` with per-attacker timing and resume
logic keyed on `(attacker, node, budget)`. Attacks need the victim checkpoint, so train the retrievers
first (see Installation).

| Datasets (`dataset=`) | Retrievers (`retriever=`) |
|---|---|
| `pubmed`, `amazon_photo`, `coauthor_cs` | `epagcl`, `gca`, `grace` |

Add-edge baseline names: `random degree pagerank infmax viking rlp graphmi dice_add spac_add pga_add
agpoa_selfdrift agpoa_querydrift nag_r_selfdrift nag_r_querydrift`.

## Per-dataset configuration

The query radius `r` and surrogate settings are selected once on a held-out validation split of targets,
disjoint from the evaluation targets; the surrogate training subgraph excludes both validation and test
targets and their incident edges, so the choice is leakage-free and uses no victim signal.

| Dataset | d | query radius `r` |
|---|---|---|
| PubMed | 512 | 1-hop |
| Amazon-Photo | 256 | blend |
| Coauthor-CS | 128 | 1-hop |

## Training retrievers

```bash
# GCA
cd src/retrievers/GCA/ && python train.py --dataset PubMed
# EPAGCL
python src/retrievers/EPAGCL/main.py --dataset PubMed
# GRACE
python src/retrievers/GRACE/train.py --dataset PubMed
```

Training is seeded for determinism, so retrained weights reproduce the checkpoints used in the paper.

## Datasets and victims

All nine graphs are treated as undirected. Edge counts are undirected and `deg` is the average node degree.

| Dataset | Nodes | Edges | Feat. | Classes | deg |
|---|---|---|---|---|---|
| PubMed | 19,717 | 44,324 | 500 | 3 | 4.5 |
| Amazon-Photo | 7,650 | 119,081 | 745 | 8 | 31.1 |
| Coauthor-CS | 18,333 | 81,894 | 6,805 | 15 | 8.9 |

**Victim retrievers** (frozen, unsupervised graph contrastive encoders): EPAGCL, GCA, GRACE.

## Extending the framework or adding a new attack

Attacks live behind a small contract in `src/attackers/base.py`. For edge addition, implement
`score_candidates(target_node) -> [N] tensor` (or override `_attack_add` for set-selection methods); the
base `attack(..., add_edge=True)` rounds to the top-`budget` candidates and inserts them. Drop a module
under `src/attackers/contrib/` exposing an `ATTACKERS` dict (or calling `register_attacker` at import),
add a `conf/attacker/<name>.yaml` with a matching `name:`, and select it with
`attacker@attackers.1=<name>`. Undercut (`src/attackers/contrib/undercut.py`) is implemented exactly
this way and doubles as a worked example.

## Ethics and responsible use

This is a research tool for studying the **robustness** of graph-based retrieval. Rank demotion through
plausible graph edits can reduce the visibility of papers, products, experts, or accounts, and can be
hard to detect because the system keeps returning coherent results. We release this implementation only
for reproducible research, controlled robustness assessment, and the development of defenses (structural
stress testing, monitoring anomalous edge insertions around high-value entities, provenance tracking for
graph edits, and audits of large rank changes after updates). Do not use it against retrieval systems you
do not own or have permission to test. All experiments here use public academic graph datasets.


## License

Released under the **Apache-2.0** license. See [LICENSE](LICENSE).
