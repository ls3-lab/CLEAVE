"""Entry point for running attacks on retrievers."""

import hydra
from omegaconf import DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig
import torch
from tqdm import tqdm
import pandas as pd
import time
import gc
import os
import json
import tempfile
import subprocess

from src.data.loader import get_dataset
from src.data.splits import load_or_create_attack_split
from src.retrievers import get_retriever
from src.attackers import load_attackers
from src.evaluations.retrieval import evaluate_retrieval, evaluation_summary
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def git_sha():
    """Best-effort 40-char commit SHA of the repo that produced a run ('unknown' if unavailable)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _atomic_to_csv(df, path):
    """Write a CSV atomically (tmp in same dir + os.replace), so a preemption mid-write can never
    truncate an existing results file — required for resumable runs."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set seed
    torch.manual_seed(cfg.general.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info('Working directory: {}\n'.format(output_dir))
    logger.info('Setting up the system for retrieval attack\nDataset: {}\nRetriever: {}\nBudget: {}\nDevice: {}\n'.format(cfg.dataset.name, cfg.retriever.name, cfg.general.budget, cfg.general.device))
    # Setup
    attackers_configs = []
    with open_dict(cfg):
        for attacker in list(cfg.attackers.values()):
            attackers_configs.append(attacker)
    device = cfg.general.device if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(cfg.dataset).to(device)
    data = dataset[0]
    _, target_nodes = load_or_create_attack_split(
        data, cfg.dataset.name, cfg.general.cache_dir, seed=cfg.general.seed)
    logger.info('Total number of nodes for attack (TEST split): {}\n'.format(len(target_nodes)))
    # VICTIM (eval target) and SURROGATE (attacker gradients) are DIFFERENT checkpoints by design
    # (black-box transfer): attacks distill/optimize on the surrogate, then transfer to the victim that
    # the metric scores against. White-box attackers (use_victim_retriever: true, e.g. graphmi) get the
    # victim instead — see load_attackers.
    retriever = get_retriever(cfg.retriever, cfg, data, device, use_oracle=False)          # victim (eval)
    surrogate_retriever = get_retriever(cfg.retriever, cfg, data, device, use_oracle=True)  # surrogate (attack)
    logger.info('Retriever: {} (victim=eval, surrogate=attack)\n'.format(cfg.retriever.name))
    attackers = load_attackers(attackers_configs,
                               system_config=cfg,
                               data=data,
                               device=device,
                               seed=cfg.general.seed,
                               retriever=surrogate_retriever,
                               victim_retriever=retriever)
    logger.info('Attackers: {}\n'.format([attacker.name for attacker in attackers]))
    # Run attacks
    logger.info('Running attacks...')
    # Results go to a STABLE dir (NOT the timestamped Hydra run dir) so the resume logic below finds
    # prior rows across job restarts, and a time-limited/preempted task continues instead of redoing
    # everything in a fresh dir (which would let a >walltime task loop forever).
    stable_out = os.path.join(cfg.general.output_dir, "attacks")
    os.makedirs(stable_out, exist_ok=True)
    results_path_name = f"{stable_out}/results_{cfg.dataset.name}_{cfg.retriever.name}_{cfg.general.budget}.csv"
    timing_path_name = results_path_name.replace('.csv', '_timing.csv')
    # RESUME: a results CSV may already hold completed (attacker, node, budget) rows (a prior run
    # that was preempted). Reload it, skip those triples, append only NEW rows, write atomically.
    prev_df = None
    done = set()
    if os.path.exists(results_path_name):
        prev_df = pd.read_csv(results_path_name)
        done = {(str(a), int(n), int(b))
                for a, n, b in zip(prev_df['attacker'], prev_df['node_id'], prev_df['budget'])}
        logger.info(f"Resume: {len(done)} (attacker,node,budget) rows already in {results_path_name}")
    new_rows = []

    def _save_results():
        frames = [f for f in (prev_df, pd.DataFrame(new_rows)) if f is not None and len(f)]
        if frames:
            _atomic_to_csv(pd.concat(frames, ignore_index=True), results_path_name)

    error_set = set()
    error_counts = {}  # attacker.name -> number of (node, budget) cases skipped due to errors
    timing_rows = []
    for attacker in attackers:
        # Release the caching allocator's reserved/fragmented pool at each attacker
        # boundary. Without this, a heavy attacker (e.g. agpoa's per-node gradient
        # loops) leaves a fragmented reserved pool that a later attacker (nettack's
        # dense-adj clones, graphmi's dense N×N) cannot allocate into on large graphs,
        # OOMing on tiny allocations. Per-attacker cleanup gives each a clean ~80GB.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        attacker_start_time = time.perf_counter()
        for node in tqdm(target_nodes, desc=f"{attacker.name} attack on {cfg.dataset.name} with {cfg.retriever.name}"):
            for budget in range(1, cfg.general.budget + 1):
                if (attacker.name, int(node), budget) in done:   # already computed (resume)
                    continue
                try:
                    attacked_data = attacker.attack(node, budget)
                    result = evaluate_retrieval(
                        retriever=retriever,
                        original_data=data,
                        attacked_data=attacked_data,
                        target_node=node,
                        attacker_name=attacker.name,
                        dataset_name=cfg.dataset.name,
                        retriever_name=cfg.retriever.name,
                        budget=budget,
                        query_radius=cfg.dataset.query_radius,
                    )
                    new_rows.append(result.__dict__)
                except Exception as e:
                    error_counts[attacker.name] = error_counts.get(attacker.name, 0) + 1
                    logger.warning(f"[skip] {attacker.name} node {node} budget {budget}: {e}")
                    if str(e) not in error_set:
                        logger.exception(f"Error in {attacker.name} attack on node {node} with budget {budget}: {e}")
                        error_set.add(str(e))
                    
    
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        attacker_inference_time = time.perf_counter() - attacker_start_time
        retriever_surrogate_time = float(getattr(attacker, 'retriever_surrogate_time', 0.0))
        timing_rows.append({
            'attacker': attacker.name,
            'retriever_name': cfg.retriever.name,
            'time_in_inference': attacker_inference_time,
            'retriever_surrogate_time': retriever_surrogate_time,
            'total_time_with_retriever_surrogate': attacker_inference_time + retriever_surrogate_time,
        })

        # Save results incrementally + atomically after each attacker (resume-safe checkpoint)
        _save_results()
        pd.DataFrame(timing_rows).to_csv(timing_path_name, index=False)
    df = pd.concat([f for f in (prev_df, pd.DataFrame(new_rows)) if f is not None and len(f)],
                   ignore_index=True)
    summary = evaluation_summary(df)
    summary.to_csv(results_path_name.replace('.csv', '_summary.csv'))
    logger.info(f"Saved summary to {results_path_name.replace('.csv', '_summary.csv')}")
    logger.info(f"Saved {len(df)} results ({len(new_rows)} new this run) to {results_path_name}")
    logger.info(f"Saved timing to {timing_path_name}")
    meta_path = results_path_name.replace('.csv', '_meta.json')
    with open(meta_path, "w") as f:
        json.dump({
            "git_sha": git_sha(),
            "seed": int(cfg.general.seed),
            "dataset": cfg.dataset.name,
            "retriever": cfg.retriever.name,
            "budget": int(cfg.general.budget),
            "num_test_targets": len(target_nodes),
        }, f, indent=2)
    logger.info(f"Saved provenance to {meta_path}")
    if error_counts:
        total_skipped = sum(error_counts.values())
        logger.warning(
            f"Skipped {total_skipped} (node,budget) evaluations due to errors; per-attacker "
            f"counts: {error_counts}. These are ABSENT from the results CSV, so affected "
            f"attackers' averages are computed over a different node set."
        )
    if error_set:
        logger.info(f"Error set: {error_set}")


if __name__ == "__main__":
    main()