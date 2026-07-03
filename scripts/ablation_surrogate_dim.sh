#!/bin/bash
# Sensitivity to the surrogate embedding dimension d in {128, 256, 512} (Fig. 4),
# on the three representative datasets and three victim retrievers.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

for ds in pubmed amazon_photo coauthor_cs; do
  for ret in epagcl gca grace; do
    for dim in 128 256 512; do
      echo "[dim=$dim] $ds / $ret"
      $PY retriever_attack.py --config-name config_cleave \
        dataset="$ds" retriever="$ret" general.budget=5 \
        attackers.1.surrogate_hidden="$dim" attackers.1.surrogate_out="$dim"
    done
  done
done
