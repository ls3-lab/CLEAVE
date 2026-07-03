import torch
import random
import numpy as np
import pickle
import os
import multiprocessing as mp
from functools import partial
from tqdm.notebook import tqdm
from tqdm import tqdm
from src.evaluations.retrieval import evaluate_retrieval
from src.attckers.victim_models import get_victim_embedding, get_victim_model


def get_2hop_non_neighbors(data, target_node):
    """
    Get nodes that are in 2-hop neighborhood of target_node but NOT directly connected.
    
    Args:
        data: Graph data with edge_index
        target_node: The target node to find 2-hop non-neighbors for
        
    Returns:
        List of node indices that are 2-hop away but not directly connected
    """
    edge_index = data.edge_index
    device = edge_index.device
    
    # Get 1-hop neighbors (directly connected nodes)
    mask_1hop = (edge_index[0] == target_node) | (edge_index[1] == target_node)
    edges_1hop = edge_index[:, mask_1hop]
    neighbors_1hop = set()
    for i in range(edges_1hop.size(1)):
        src, dst = edges_1hop[0, i].item(), edges_1hop[1, i].item()
        if src == target_node:
            neighbors_1hop.add(dst)
        else:
            neighbors_1hop.add(src)
    
    # Get 2-hop neighbors (neighbors of neighbors)
    neighbors_2hop = set()
    for neighbor in neighbors_1hop:
        mask_2hop = (edge_index[0] == neighbor) | (edge_index[1] == neighbor)
        edges_2hop = edge_index[:, mask_2hop]
        for i in range(edges_2hop.size(1)):
            src, dst = edges_2hop[0, i].item(), edges_2hop[1, i].item()
            if src == neighbor:
                neighbors_2hop.add(dst)
            else:
                neighbors_2hop.add(src)
    
    # Remove target node and 1-hop neighbors from 2-hop set
    # These are nodes that are exactly 2 hops away (not connected directly)
    non_neighbors_2hop = neighbors_2hop - neighbors_1hop - {target_node}
    
    return list(non_neighbors_2hop)


def similarity(num_nodes, query_embedding, dataset_embeddings):
    """Calculate cosine similarity and return top-k indices."""
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding, dataset_embeddings, dim=1
    )
    top_k_indices = torch.argsort(similarities, descending=True)[:num_nodes]
    return top_k_indices


def get_position(node, top_k_indices):
    """Get the position of a node in the ranking."""
    position = (top_k_indices == node).nonzero(as_tuple=True)[0]
    if len(position) == 0:
        return -1
    return position[0].item()


def compute_position_change(node, old_ranking, new_ranking):
    """Compute position change for a node between two rankings."""
    old_pos = get_position(node, old_ranking)
    new_pos = get_position(node, new_ranking)
    return new_pos - old_pos  # Positive means demoted, negative means promoted


def calculate_add_edge_performance_all_data(args, data, graph_model, data_name, trained_models_path, main_seed, batch_size=1): 
    """
    Calculate the effect of adding edges from 2-hop non-neighbors to the target node.
    For each target node, find nodes in 2-hop neighborhood that are not connected,
    then add edges one by one to see the effect on ranking.
    """
    
    torch.manual_seed(main_seed)
    torch.cuda.manual_seed(main_seed)
    np.random.seed(main_seed)
    random.seed(main_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vitim_model = args.victim
    vitim_model_lower = vitim_model.lower()
    args.model = graph_model
    args.data_name = data_name
    args.dataset = data_name

    data = data.to(device)
    num_nodes = data.x.size(0)
    print(vitim_model_lower)
    
    model = get_victim_model(args, data, data_name, graph_model, device)

    h = get_victim_embedding(vitim_model_lower, model, data.x, data.edge_index, device)
    dataset_embeddings = h
    edge_performance = {}
    

    file_path = os.path.join(trained_models_path, f'{vitim_model}_Full_{data_name}_{graph_model}_add_edge_performance.pkl')
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            edge_performance = pickle.load(file)

 
    for node in tqdm(range(num_nodes), desc='Add Edge Performance {} [{}-{}]'.format(vitim_model, graph_model, data_name)):
        if node in edge_performance:
            continue
        
        edge_performance[node] = {}

        # Get 2-hop non-neighbors (potential edges to add)
        candidate_nodes = get_2hop_non_neighbors(data, node)
        
        if len(candidate_nodes) == 0:
            # No candidates to add edges to
            with open(file_path, "wb") as file:
                pickle.dump(edge_performance, file)
            continue

        # Isolate the node by removing all its edges (Query)
        mask = (data.edge_index[0] != node) & (data.edge_index[1] != node)
        isolated_edge_index = data.edge_index[:, mask]

        isolated_embeddings = get_victim_embedding(vitim_model_lower, model, data.x, isolated_edge_index, device)
        target_node_embedding = isolated_embeddings[node].unsqueeze(0)

        top_k_indices = similarity(num_nodes, target_node_embedding, dataset_embeddings)
        init_position = get_position(node, top_k_indices)


        for candidate in candidate_nodes:
            new_data = data.clone()
            
            # Add edge (node, candidate) and (candidate, node) for undirected graph
            new_edge = torch.tensor([[node, candidate], [candidate, node]], dtype=torch.long, device=device)
            new_data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

            updated_embeddings = get_victim_embedding(vitim_model_lower, model, 
                                                      new_data.x, new_data.edge_index, 
                                                      device)

            # Get query embedding with the new edge added but target node isolated
            attacked_mask = (new_data.edge_index[0] != node) & (new_data.edge_index[1] != node)
            attacked_isolated_edge_index = new_data.edge_index[:, attacked_mask]

            attacked_isolated_embeddings = get_victim_embedding(vitim_model_lower, model, 
                                                      new_data.x, attacked_isolated_edge_index, 
                                                      device)
            new_query_embedding = attacked_isolated_embeddings[node].unsqueeze(0)


            new_top_k_indices = similarity(num_nodes, new_query_embedding, updated_embeddings)

            avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items = [None] * 5
            position_change = compute_position_change(node, top_k_indices, new_top_k_indices)
            edge_performance[node].update({(node, candidate): [None, None, [position_change, avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items, init_position]]})
            
        
        with open(file_path, "wb") as file:
            pickle.dump(edge_performance, file)
    
    print(f"Add edge performance for all nodes is saved at: {file_path}")
    
    return edge_performance


def calculate_add_edge_performance_all_data_batched(args, data, graph_model, data_name, trained_models_path, main_seed, batch_size=8):
    """
    Batched version that processes multiple nodes in parallel.
    
    Args:
        batch_size: Number of nodes to process simultaneously (default: 8)
    """
    torch.manual_seed(main_seed)
    torch.cuda.manual_seed(main_seed)
    np.random.seed(main_seed)
    random.seed(main_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vitim_model = args.victim
    vitim_model_lower = vitim_model.lower()
    args.model = graph_model
    args.data_name = data_name
    args.dataset = data_name

    data = data.to(device)
    num_nodes = data.x.size(0)
    print(f"Processing with batch_size={batch_size}")
    print(vitim_model_lower)
    model = get_victim_model(args, data, data_name, graph_model, device)
    
    # Get base embeddings
    h = get_victim_embedding(vitim_model_lower, model, data.x, data.edge_index, device)
    dataset_embeddings = h
    edge_performance = {}
    
    file_path = os.path.join(trained_models_path, f'{vitim_model}_Full_{data_name}_{graph_model}_add_edge_performance.pkl')
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            edge_performance = pickle.load(file)

    # Get list of nodes to process
    nodes_to_process = [node for node in range(num_nodes) if node not in edge_performance]
    
    if not nodes_to_process:
        print("All nodes already processed!")
        return edge_performance
    
    # Process nodes in batches
    num_batches = (len(nodes_to_process) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f'Batch Processing [{vitim_model}-{graph_model}-{data_name}]'):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(nodes_to_process))
        batch_nodes = nodes_to_process[batch_start:batch_end]
        
        # Process this batch of nodes
        batch_results = process_node_batch(batch_nodes, data, model, vitim_model_lower, 
                                          dataset_embeddings, num_nodes, device)
        
        # Update edge_performance with batch results
        edge_performance.update(batch_results)
        
        # Save checkpoint after each batch
        with open(file_path, "wb") as file:
            pickle.dump(edge_performance, file)
    
    print(f"Add edge performance for all nodes saved at: {file_path}")
    return edge_performance


def process_node_batch(batch_nodes, data, model, vitim_model_lower, dataset_embeddings, num_nodes, device):
    """
    Process a batch of nodes in parallel.
    
    Returns:
        Dictionary mapping node_id -> edge_performance for the batch
    """
    batch_results = {}
    
    # Step 1: Get isolated embeddings for all nodes in batch
    isolated_embeddings_batch = []
    candidate_nodes_batch = []
    init_positions_batch = []
    
    for node in batch_nodes:
        # Get 2-hop non-neighbors
        candidate_nodes = get_2hop_non_neighbors(data, node)
        candidate_nodes_batch.append(candidate_nodes)
        
        if len(candidate_nodes) == 0:
            isolated_embeddings_batch.append(None)
            init_positions_batch.append((None, None))
            continue
        
        # Isolate the node
        mask = (data.edge_index[0] != node) & (data.edge_index[1] != node)
        isolated_edge_index = data.edge_index[:, mask]
        
        isolated_embeddings = get_victim_embedding(vitim_model_lower, model, data.x, isolated_edge_index, device)
        target_node_embedding = isolated_embeddings[node].unsqueeze(0)
        isolated_embeddings_batch.append(target_node_embedding)
        
        # Get initial position
        top_k_indices = similarity(num_nodes, target_node_embedding, dataset_embeddings)
        init_position = get_position(node, top_k_indices)
        init_positions_batch.append((top_k_indices, init_position))
    
    # Step 2: Process edges for each node
    for idx, node in enumerate(batch_nodes):
        batch_results[node] = {}
        candidate_nodes = candidate_nodes_batch[idx]
        
        if len(candidate_nodes) == 0:
            continue
            
        target_node_embedding = isolated_embeddings_batch[idx]
        top_k_indices, init_position = init_positions_batch[idx]
        
        # Process each candidate edge for this node
        for candidate in candidate_nodes:
            new_data = data.clone()
            
            # Add edge
            new_edge = torch.tensor([[node, candidate], [candidate, node]], dtype=torch.long, device=device)
            new_data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
            
            # Get updated embeddings
            updated_embeddings = get_victim_embedding(vitim_model_lower, model, 
                                                      new_data.x, new_data.edge_index, device)
            
            # Get attacked isolated embeddings
            attacked_mask = (new_data.edge_index[0] != node) & (new_data.edge_index[1] != node)
            attacked_isolated_edge_index = new_data.edge_index[:, attacked_mask]
            
            attacked_isolated_embeddings = get_victim_embedding(vitim_model_lower, model, 
                                                      new_data.x, attacked_isolated_edge_index, device)
            new_query_embedding = attacked_isolated_embeddings[node].unsqueeze(0)
            
            # Calculate similarity and metrics
            new_top_k_indices = similarity(num_nodes, new_query_embedding, updated_embeddings)
            
            avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items = [None] * 5
            position_change = compute_position_change(node, top_k_indices, new_top_k_indices)
            
            batch_results[node].update({
                (node, candidate): [None, None, [position_change, avg_promoted, avg_demoted, 
                                        avg_changed, avg_unchanged, total_items, init_position]]
            })
    
    return batch_results


def process_single_batch_worker(batch_idx, batch_size, args, data, graph_model, data_name, 
                                 trained_models_path, main_seed, vitim_model, num_nodes, device_id=0):
    """
    Worker function to process a single batch. Designed to be run in parallel.
    Saves results to a separate file per batch.
    
    Args:
        batch_idx: Index of the batch to process
        batch_size: Number of nodes per batch
        device_id: GPU device ID to use (default: 0)
    
    Returns:
        Tuple of (batch_idx, success, file_path)
    """
    try:
        # Set device for this worker
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        # Set seeds for reproducibility
        worker_seed = main_seed + batch_idx
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        vitim_model_lower = vitim_model.lower()
        args.model = graph_model
        args.data_name = data_name
        args.dataset = data_name
        
        # Move data to device
        data = data.to(device)
        
        # Create batch-specific filename
        batch_file_path = os.path.join(
            trained_models_path, 
            f'{vitim_model}_Full_{data_name}_{graph_model}_add_edge_batch_{batch_idx}.pkl'
        )
        
        # Skip if batch already processed
        if os.path.exists(batch_file_path):
            print(f"Batch {batch_idx} already processed, skipping...")
            return (batch_idx, True, batch_file_path)
        
        # Load model
        model = get_victim_model(args, data, data_name, graph_model, device)
        
        # Get base embeddings
        h = get_victim_embedding(vitim_model_lower, model, data.x, data.edge_index, device)
        dataset_embeddings = h
        
        # Calculate which nodes this batch should process
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_nodes)
        batch_nodes = list(range(batch_start, batch_end))
        
        if not batch_nodes:
            return (batch_idx, True, None)
        
        print(f"Worker processing batch {batch_idx}: nodes {batch_start}-{batch_end-1} on {device}")
        
        # Process this batch
        batch_results = process_node_batch(
            batch_nodes, data, model, vitim_model_lower, 
            dataset_embeddings, num_nodes, device
        )
        
        # Save batch results to separate file
        with open(batch_file_path, "wb") as file:
            pickle.dump(batch_results, file)
        
        print(f"Batch {batch_idx} completed and saved to {batch_file_path}")
        return (batch_idx, True, batch_file_path)
        
    except Exception as e:
        print(f"Error processing batch {batch_idx}: {e}")
        import traceback
        traceback.print_exc()
        return (batch_idx, False, None)


def calculate_add_edge_performance_multiprocess(args, data, graph_model, data_name, 
                                           trained_models_path, main_seed, 
                                           batch_size=8, num_processes=4, device_ids=None):
    """
    Multi-process batch processing version. Each process handles different batches in parallel.
    
    Args:
        batch_size: Number of nodes per batch
        num_processes: Number of parallel processes to use
        device_ids: List of GPU device IDs to use (e.g., [0, 1, 2, 3]). If None, uses device 0 for all.
    
    Returns:
        Path to the merged final file
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass
    
    vitim_model = args.victim
    
    # Move data to CPU for multiprocessing (workers will move to GPU)
    data = data.cpu()
    num_nodes = data.x.size(0)
    
    # Calculate total number of batches
    num_batches = (num_nodes + batch_size - 1) // batch_size
    
    print(f"="*80)
    print(f"Multi-Process Batch Processing (Add Edge)")
    print(f"Dataset: {data_name}, Model: {graph_model}, Victim: {vitim_model}")
    print(f"Total nodes: {num_nodes}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {num_batches}")
    print(f"Parallel processes: {num_processes}")
    print(f"="*80)
    
    # Setup device IDs
    if device_ids is None:
        device_ids = [0] * num_processes
    else:
        # Cycle through provided device IDs if fewer than num_processes
        device_ids = [device_ids[i % len(device_ids)] for i in range(num_processes)]
    
    # Create subfolder for batch files
    batch_folder = os.path.join(trained_models_path, f'{data_name}_{graph_model}_{vitim_model}_add_edge')
    os.makedirs(batch_folder, exist_ok=True)
    print(f"Batch files will be saved to: {batch_folder}")
    
    # Check which batches are already processed
    batches_to_process = []
    for batch_idx in range(num_batches):
        batch_file_path = os.path.join(
            batch_folder,
            f'{vitim_model}_Full_{data_name}_{graph_model}_add_edge_batch_{batch_idx}.pkl'
        )
        if not os.path.exists(batch_file_path):
            batches_to_process.append(batch_idx)
    
    if not batches_to_process:
        print("All batches already processed! Proceeding to merge...")
    else:
        print(f"Batches to process: {len(batches_to_process)} out of {num_batches}")
        print(f"Batch indices: {batches_to_process[:10]}{'...' if len(batches_to_process) > 10 else ''}")
        
        # Create worker function with fixed parameters
        worker_func = partial(
            process_single_batch_worker,
            batch_size=batch_size,
            args=args,
            data=data,
            graph_model=graph_model,
            data_name=data_name,
            trained_models_path=batch_folder,
            main_seed=main_seed,
            vitim_model=vitim_model,
            num_nodes=num_nodes
        )
        
        # Process batches in parallel
        print(f"\nStarting parallel processing with {num_processes} processes...")
        
        # Use multiprocessing Pool
        with mp.Pool(processes=num_processes) as pool:
            # Assign device IDs to batches in round-robin fashion
            batch_device_pairs = [(batch_idx, device_ids[i % len(device_ids)]) 
                                  for i, batch_idx in enumerate(batches_to_process)]
            
            # Process batches with progress bar
            results = []
            for batch_idx, device_id in tqdm(batch_device_pairs, desc="Processing batches"):
                result = pool.apply_async(worker_func, (batch_idx,), {'device_id': device_id})
                results.append(result)
            
            # Wait for all results
            pool.close()
            pool.join()
            
            # Collect results
            completed = 0
            failed = 0
            for result in results:
                batch_idx, success, file_path = result.get()
                if success:
                    completed += 1
                else:
                    failed += 1
            
            print(f"\nProcessing complete: {completed} batches succeeded, {failed} batches failed")
    
    # Merge all batch files
    print(f"\nMerging batch files...")
    final_file_path = merge_batch_files(vitim_model, data_name, graph_model, trained_models_path, num_batches, batch_folder)
    
    return final_file_path


def merge_batch_files(vitim_model, data_name, graph_model, trained_models_path, num_batches, batch_folder=None):
    """
    Merge all batch files into a single add_edge_performance file.
    
    Args:
        vitim_model: Victim model name
        data_name: Dataset name
        graph_model: Graph model name
        trained_models_path: Path to save final merged file
        num_batches: Total number of batches
        batch_folder: Folder containing batch files (if None, uses trained_models_path)
    
    Returns:
        Path to the merged file
    """
    merged_results = {}
    
    # Use batch_folder if provided, otherwise use trained_models_path
    if batch_folder is None:
        batch_folder = os.path.join(trained_models_path, f'{data_name}_{graph_model}_{vitim_model}_add_edge')
    
    print(f"Merging {num_batches} batch files from: {batch_folder}")
    
    for batch_idx in tqdm(range(num_batches), desc="Merging batches"):
        batch_file_path = os.path.join(
            batch_folder,
            f'{vitim_model}_Full_{data_name}_{graph_model}_add_edge_batch_{batch_idx}.pkl'
        )
        
        if os.path.exists(batch_file_path):
            try:
                with open(batch_file_path, "rb") as file:
                    batch_data = pickle.load(file)
                    merged_results.update(batch_data)
            except Exception as e:
                print(f"Warning: Could not load batch {batch_idx}: {e}")
        else:
            print(f"Warning: Batch file {batch_idx} not found: {batch_file_path}")
    
    # Save merged results
    final_file_path = os.path.join(
        trained_models_path,
        f'{vitim_model}_Full_{data_name}_{graph_model}_add_edge_performance.pkl'
    )
    
    with open(final_file_path, "wb") as file:
        pickle.dump(merged_results, file)
    
    print(f"\nMerge complete!")
    print(f"Total nodes in merged file: {len(merged_results)}")
    print(f"Merged file saved to: {final_file_path}")
    
    return final_file_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcn")
    parser.add_argument("--data_name", type=str, default="Cora")
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eval_optim', type=str, default='Adam')
    parser.add_argument('--eval_lr', type=float, default=0.01)
    parser.add_argument('--eval_weight_decay', type=float, default=0.0)
    parser.add_argument('--eval_epoch', type=int, default=1000)
    parser.add_argument('--output_path', type=str, default='./tmp')

    args = parser.parse_args()
