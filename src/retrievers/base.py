import torch
from omegaconf import DictConfig
from torch_geometric.data import Data
from src.utils.logger_utils import setup_logger
import abc

logger = setup_logger(__name__)

class Retriever(abc.ABC):
    def __init__(self, cfg: DictConfig, system_cfg: DictConfig, data: Data, device, use_oracle: bool = False):
        self.cfg = cfg
        self.system_cfg = system_cfg
        self.data = data
        self.device = device
        self.best_classifier = None
        self.use_oracle = use_oracle
        self.load_model()


    def get_attack_model_path(self):
        path = self.cfg.attack_model_path[self.system_cfg.dataset.name][self.system_cfg.general.graph_model]
        logger.info(f"Attack model path: {path}")
        return path

    def get_oracle_model_path(self):
        path = self.cfg.oracle_model_path[self.system_cfg.dataset.name][self.system_cfg.general.graph_model]
        logger.info(f"Oracle model path: {path}")
        return path

    def get_model_path(self):
        if self.use_oracle:
            return self.get_oracle_model_path()
        return self.get_attack_model_path()

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        self.model.eval()
        edge_index = edge_index.to(self.device)
        x = x.to(self.device) 
        with torch.no_grad():
            return self.model(x, edge_index)

    def retrieve(self, node_embedding: torch.Tensor, embedding):
        first_embedding = node_embedding.to(self.device)
        second_embedding = embedding.to(self.device)
        cosine_sim = torch.nn.functional.cosine_similarity(first_embedding.unsqueeze(1), second_embedding.unsqueeze(0), dim=-1)
        return torch.argsort(cosine_sim, dim=1, descending=True, stable=True)

    def get_node_embedding(self, node_id: int, embedding):
        return embedding[node_id].unsqueeze(0)
            

    def get_class(self, embedding: torch.Tensor):
        embedding = embedding.to(self.device)
        if self.best_classifier is None:
            self.best_classifier = self.class_mlp(embedding, self.data.y)
        self.best_classifier.eval()
        with torch.no_grad():
            output = self.best_classifier(embedding)
            _, class_predictions = torch.max(output, 1)
            return class_predictions.item()

    @abc.abstractmethod
    def load_model(self):
        pass