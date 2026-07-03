from src.retrievers.models import EPAGCLRetriever, GCARetriever, GRACERetriever
from omegaconf import DictConfig
from torch_geometric.data import Data


__all__ = [
    'EPAGCLRetriever',
    'GCARetriever',
    'GRACERetriever'
]

def get_retriever(retriever_cfg: DictConfig, cfg: DictConfig, data: Data, device, use_oracle: bool = False):
    retriever_name = retriever_cfg.name.lower()
    retriever_models = {
        'epagcl': EPAGCLRetriever,
        'gca': GCARetriever,
        'grace': GRACERetriever
    }
    return retriever_models[retriever_name](retriever_cfg, cfg, data, device, use_oracle)