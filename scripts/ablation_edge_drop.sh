#!/bin/bash
# Sensitivity to the contrastive augmentation edge-drop rate (Fig. 5), on the
# three representative datasets and three victim retrievers. Rate 0.0 disables
# augmentation.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

for ds in pubmed amazon_photo coauthor_cs; do
  for ret in epagcl gca grace; do
    for p in 0.0 0.1 0.2 0.3 0.4 0.5; do
      echo "[edge-drop=$p] $ds / $ret"
      $PY retriever_attack.py --config-name config_cleave \
        dataset="$ds" retriever="$ret" general.budget=5 \
        attackers.1.drop1="$p" attackers.1.drop2="$p"
    done
  done
done
