import torch
import torch.nn.functional as F


def compute_avg_rank_change(model, data, val_mask, device, encode_fn=None):
    """Compute average node rank change on the validation set using cosine similarity.

    For each validation node:
      1. Compute dataset embeddings on the full graph.
      2. Isolate the node (remove all its edges) and recompute embeddings.
      3. Use the isolated node's embedding as a query; rank all nodes by
         cosine similarity against the full-graph embeddings.
      4. Record where the node itself appears in this ranking (its "position").
      5. The rank change is ``isolated_position - 0`` (ideal position is rank 0,
         i.e. the node is most similar to its own full-graph embedding).

    The model with the **minimum** average rank change is the most stable
    (immune to edge perturbation) and should be saved as the best checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        The GCL encoder model (EPAGCL / GCA / GRACE).  Must accept
        ``(x, edge_index)`` and return node embeddings.
    data : torch_geometric.data.Data
        The graph data object (with ``x``, ``edge_index``, ``y``).
    val_mask : torch.BoolTensor
        Boolean mask of shape ``(num_nodes,)`` selecting validation nodes.
    device : torch.device | str
        Device for computation.
    encode_fn : callable, optional
        Custom encoding function ``encode_fn(model, x, edge_index) -> embeddings``.
        If *None*, defaults to ``model(x, edge_index)``.  For EPAGCL whose
        forward returns ``(embedding, projection)``, pass a lambda that
        extracts only the embedding.

    Returns
    -------
    avg_rank_change : float
        Mean rank (position) of each validation node in the cosine-similarity
        ranking.  Lower is better.
    """
    model.eval()
    val_indices = val_mask.nonzero(as_tuple=True)[0]

    if len(val_indices) == 0:
        return float('inf')

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # --- Full-graph embeddings (used as the "database") --------------------
    with torch.no_grad():
        if encode_fn is not None:
            dataset_embeddings = encode_fn(model, x, edge_index)
        else:
            dataset_embeddings = model(x, edge_index)

    dataset_embeddings = F.normalize(dataset_embeddings, dim=-1)

    total_rank = 0.0

    for node in val_indices:
        node_item = node.item()
        # Isolate the node: remove every edge touching it
        mask = (edge_index[0] != node_item) & (edge_index[1] != node_item)
        isolated_edge_index = edge_index[:, mask]

        with torch.no_grad():
            if encode_fn is not None:
                isolated_embeddings = encode_fn(model, x, isolated_edge_index)
            else:
                isolated_embeddings = model(x, isolated_edge_index)

        query = F.normalize(isolated_embeddings[node_item].unsqueeze(0), dim=-1)

        # Cosine similarity between the query and all full-graph embeddings
        cos_sim = torch.mm(query, dataset_embeddings.t()).squeeze(0)  # (N,)

        # Rank: descending similarity → position 0 = most similar
        ranking = torch.argsort(cos_sim, descending=True)
        position = (ranking == node_item).nonzero(as_tuple=True)[0].item()

        total_rank += position

    avg_rank_change = total_rank / len(val_indices)
    return avg_rank_change
