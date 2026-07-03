import torch
import statistics
from dataclasses import dataclass
from torch_geometric.data import Data

from src.oracle.query import neighborhood_centroid_query

@dataclass
class RetrievalResult:
    """Result of a single retrieval attack evaluation."""
    node_id: int
    budget: int
    attacker: str
    dataset_name: str
    retriever_name: str
    init_position: int
    final_positions: list
    position_change: list
    edges_removed: list
    avg_final_position: int
    avg_position_change: int
    std_final_position: float
    std_position_change: float


def _removed_edges(original_data: Data, attacked_data: Data) -> list:
    """Return directed edges present in original_data but missing after attack."""
    recorded_edges = getattr(attacked_data, "edges_removed", None)
    if recorded_edges is not None:
        return [tuple(map(int, edge)) for edge in recorded_edges]

    original_edge_index = original_data.edge_index
    attacked_edge_index = attacked_data.edge_index
    if original_edge_index.size(1) == attacked_edge_index.size(1):
        return []

    num_nodes = int(getattr(original_data, "num_nodes", 0) or original_data.x.size(0))
    original_codes = original_edge_index[0].to(torch.long) * num_nodes + original_edge_index[1].to(torch.long)
    attacked_codes = attacked_edge_index[0].to(torch.long) * num_nodes + attacked_edge_index[1].to(torch.long)
    removed_mask = ~torch.isin(original_codes, attacked_codes)
    removed = original_edge_index[:, removed_mask].t().detach().cpu().tolist()
    return [tuple(map(int, edge)) for edge in removed]


def evaluate_retrieval(
    retriever,
    original_data,
    attacked_data,
    target_node: int,
    attacker_name: str,
    dataset_name: str,
    retriever_name: str,
    budget: int,
    query_radius: str,
) -> RetrievalResult:
    """
    Evaluate retrieval performance before and after attack.
    
    Args:
        retriever: Victim retriever model
        original_data: Original graph
        attacked_data: Graph after attack
        target_node: Node being attacked
        attacker_name: Name of attack method
        retriever_name: Name of retriever model
        budget: Number of edges removed
        
    Returns:
        RetrievalResult with position metrics
    """
    # Get embeddings on original graph
    if not isinstance(attacked_data, list):
        attacked_data = [attacked_data]

    # FROZEN per-dataset neighborhood-centroid query (built ONCE on the clean graph, in the victim's
    # embedding space; does not move under attack). See src/oracle/query.py.
    dataset_embedding = retriever.encode(original_data.x, original_data.edge_index)
    num_nodes = int(getattr(original_data, "num_nodes", 0) or original_data.x.size(0))
    node_embedding = neighborhood_centroid_query(
        original_data.edge_index, target_node, query_radius, dataset_embedding, num_nodes)
    orig_ranking = retriever.retrieve(node_embedding, dataset_embedding)

    init_position = (orig_ranking == target_node).nonzero(as_tuple=True)[1].item()
    final_positions = []
    edges_removed = []
    # Get embeddings on attacked graph
    for attacked_data_instance in attacked_data:
        edges_removed.append(_removed_edges(original_data, attacked_data_instance))
        attack_embeddings = retriever.encode(attacked_data_instance.x, attacked_data_instance.edge_index)
        attack_ranking = retriever.retrieve(node_embedding, attack_embeddings)
        attacked_position = (attack_ranking == target_node).nonzero(as_tuple=True)[1].item() 
        final_positions.append(attacked_position)
    position_changes = [final_position - init_position for final_position in final_positions]
    avg_positions = sum(final_positions) / len(final_positions)
    avg_position_change = sum(position_changes) / len(position_changes)
    std_positions = statistics.stdev(final_positions) if len(final_positions) > 1 else 0.0
    std_position_change = statistics.stdev(position_changes) if len(position_changes) > 1 else 0.0
    return RetrievalResult(
        node_id=target_node,
        budget=budget,
        attacker=attacker_name,
        dataset_name=dataset_name,
        retriever_name=retriever_name,
        init_position=init_position,
        final_positions=final_positions,
        position_change=position_changes,
        edges_removed=edges_removed,
        avg_final_position=avg_positions,
        avg_position_change=avg_position_change,
        std_final_position=std_positions,
        std_position_change=std_position_change,
    )


def evaluation_summary(df):
    summary = df.drop(['final_positions', 'position_change', 'edges_removed'], axis=1, errors='ignore').groupby(['attacker', 'retriever_name', 'budget']).agg({
                         'avg_position_change': ['mean'],
                         'std_position_change': ['mean'],
                         'std_final_position': ['mean'],
                         'init_position': ['mean']
                         }).unstack(level='budget')
    summary = summary.rename(columns={
        'std_position_change': 'avg_std_position_change',
        'std_final_position': 'avg_std_final_position',
    }, level=0)
    return summary
