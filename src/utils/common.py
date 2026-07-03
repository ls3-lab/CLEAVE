import torch

def remove_edges(data, edges_to_remove, n_edges):
    """
    Remove the top-n edges from the data.
    """
    
    # Ensure we only remove up to n_edges if the provided list is longer
    edges_to_remove = edges_to_remove[:n_edges]
    edges_to_remove_set = set(edges_to_remove)
    edges_to_remove = list(edges_to_remove_set)
    
    # If no edges to remove, return the original data
    if len(edges_to_remove) == 0:
        print('waning... no edges to remove')
        return data.clone()
    
    # Apply mask
    # mask = torch.tensor([tuple(edge.tolist()) not in edges_to_remove_set for edge in data.edge_index.t()], dtype=torch.bool)
    edges_to_remove_tensor = torch.tensor(edges_to_remove, dtype=torch.long, device=data.edge_index.device)

    edge_index_T = data.edge_index.t()  # shape [num_edges, 2]
    mask = ~( (edge_index_T[:, None, :] == edges_to_remove_tensor[None, :, :]).all(dim=2).any(dim=1) )
    
    updated_data = data.clone()
    updated_data.edge_index = data.edge_index[:, mask]
    
    return updated_data