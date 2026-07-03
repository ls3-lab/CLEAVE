#!/bin/bash
# Run the full head-to-head suite (CLEAVE plus the twelve baselines composed in
# conf/config.yaml) across all nine datasets and three victim retrievers.
# CLEAVE is attackers.1, so the per-dataset Table III overrides apply to it.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

cfg() {  # dataset -> "drop1 drop2 epochs dim beta" (Table III)
  case "$1" in
    pubmed)           echo "0.5 0.5 600 512 0.0" ;;
    amazon_photo)     echo "0.3 0.3 300 256 0.0" ;;
    coauthor_cs)      echo "0.4 0.4 300 128 0.0" ;;
    coauthor_physics) echo "0.3 0.4 300 256 0.0" ;;
    corafull)         echo "0.3 0.4 300 256 0.0" ;;
    dblp)             echo "0.3 0.4 300 256 1.0" ;;
    amazon_computers) echo "0.3 0.4 300 256 0.0" ;;
    citeseer)         echo "0.3 0.4  50 256 0.0" ;;
    cora)             echo "0.3 0.4  50 256 1.0" ;;
  esac
}

DATASETS="cora citeseer pubmed corafull dblp amazon_photo amazon_computers coauthor_cs coauthor_physics"
for ds in $DATASETS; do
  read -r d1 d2 ep dim beta <<< "$(cfg "$ds")"
  for ret in epagcl gca grace; do
    echo "[SUITE] $ds / $ret"
    $PY retriever_attack.py dataset="$ds" retriever="$ret" general.budget=5 \
      attackers.1.drop1="$d1" attackers.1.drop2="$d2" \
      attackers.1.surrogate_epochs="$ep" \
      attackers.1.surrogate_hidden="$dim" attackers.1.surrogate_out="$dim" \
      attackers.1.degree_bias="$beta"
  done
done
