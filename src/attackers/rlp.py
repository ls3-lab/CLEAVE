import os
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, APPNP
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from omegaconf import DictConfig

from src.attackers.base import Attacker
from src.data.splits import split_data, get_target_nodes


# Define Encoder class at module level for pickling
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.linear_mu = nn.Linear(in_channels, out_channels)
        self.linear_logstd = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)
    
    def forward(self, x, edge_index):
        mu = self.linear_mu(x)
        mu = self.propagate(mu, edge_index)
        logstd = self.linear_logstd(x)
        logstd = self.propagate(logstd, edge_index)
        return mu, logstd


class RLPAttack(Attacker):
    """
    Reverse Link Prediction attack that trains a VGAE model to predict
    which edges are least likely to exist.
    """

    def __init__(self, cfg, system_config: DictConfig, data: Data, **kwargs):
        super().__init__(cfg, system_config, data)
        self.epoch = cfg.epoch 
        self.channels = cfg.channels
        self.learning_rate = cfg.learning_rate
        self.train_mask, self.val_mask, self.test_mask = split_data(data)
        self.selected_nodes = torch.tensor(get_target_nodes(self.test_mask), dtype=torch.long, device=self.device)
        self.rlp_models = {}
        self.rlp_model = None

    def _create_test_set(self, target_node: int) -> Data:
        """Create test set containing only edges connected to target node."""
        selected_node = torch.tensor([target_node], device=self.data.edge_index.device)
        mask = torch.isin(self.data.edge_index[0], selected_node) | torch.isin(self.data.edge_index[1], selected_node)
        filtered_edge_index = self.data.edge_index[:, mask]
        test_set = self.data.clone()
        test_set.edge_index = filtered_edge_index
        return test_set

    def _compute_edge_scores(self, rlp_model, target_node: int) -> dict:
        """Compute link prediction scores for edges connected to target node."""
        if rlp_model is None:
            return {}
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_set = self._create_test_set(target_node)
        test_set = NormalizeFeatures()(test_set).to(device)
        
        rlp_model.eval()
        with torch.no_grad():
            z = rlp_model.encode(test_set.x, test_set.edge_index)
            scores = rlp_model.decoder(z, test_set.edge_index)
        
        # Build dict mapping edge -> score
        edge_scores = {}
        edges = test_set.edge_index.t().cpu().tolist()
        scores = scores.cpu().tolist()
        
        for i, (src, dst) in enumerate(edges):
            edge_scores[(int(src), int(dst))] = scores[i]
        self.edge_scores = edge_scores
        return edge_scores

    def score_edges(self, target_node: int) -> dict[Tuple[int, int], float]:
        """
        Score each edge connected to target node based on link prediction.
        
        Returns:
            Dict mapping (src, dst) -> score
            Lower score = higher priority for removal
        """
        target_edges = self.get_target_edges(target_node)
        scores = {}
        
        for edge in target_edges:
            # Use precomputed edge scores, default to 0 if not found
            scores[edge] = self.edge_scores.get(edge, 0.0)
        
        return scores


    def attack(self, target_node: int, budget: int) -> Data:
        """
        Execute attack by removing top-scored edges.
        
        Returns:
            Modified graph data with edges removed
        """
        embedding_list = []
        for i in range(self.trials):
            local_seed = self.seed + i
            torch.manual_seed(local_seed)
            
            if local_seed not in self.rlp_models:
                self.rlp_models[local_seed] = self.train_rlp_model(local_seed)

            self._compute_edge_scores(self.rlp_models[local_seed], target_node)  # fills self.edge_scores (both directions)
            # Score over CANONICAL undirected edges (get_target_edges, one key per neighbor) — NOT the
            # directed decoder keys. Slicing directed keys removed (t,n) AND (n,t), which _remove_edges
            # dedups to ONE undirected edge => only ~half the budget was actually spent (audit H3).
            scores = self.score_edges(target_node)
            # ascending: lowest link-probability (most anomalous) first; budget counts undirected edges
            sorted_edges = sorted(scores.keys(), key=lambda e: scores[e])
            edges_to_remove = sorted_edges[:budget]

            embedding_list.append(self._remove_edges(edges_to_remove))
        return embedding_list 

    def train_rlp_model(self, seed):
        """Train a VGAE model for reverse link prediction."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        training_set = self.create_training_set()
        epochs = self.epoch
        channels = self.channels
        learning_rate = self.learning_rate
        # Normalize features
        training_set = NormalizeFeatures()(training_set).to(device)
        
        # Initialize VGAE model
        model = VGAE(Encoder(training_set.num_features, channels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Generate negative edges for training
        neg_train_edge_index = negative_sampling(
            training_set.edge_index, 
            num_nodes=training_set.num_nodes,
            num_neg_samples=training_set.edge_index.size(1)
        )
        
        # Training function
        def reverse_train():
            model.train()
            optimizer.zero_grad()
            z = model.encode(training_set.x, training_set.edge_index)
            loss = model.recon_loss(z, training_set.edge_index)
            loss += (1 / training_set.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            return loss
        
        # Training loop
        for epoch in tqdm(range(1, epochs + 1)):
            loss = reverse_train()
            loss = float(loss)
        
        return model


    def create_training_set(self):
        """Create training set by removing edges of selected nodes."""
        edge_index = self.data.edge_index.to(self.device)
        selected_nodes = self.selected_nodes.to(self.device)
        # Mask to keep only edges that do not involve selected nodes
        mask = ~((edge_index[0].unsqueeze(1) == selected_nodes).any(dim=1) |
                (edge_index[1].unsqueeze(1) == selected_nodes).any(dim=1))

        # Create training set by removing edges of selected nodes
        training_edge_index = edge_index[:, mask]

        # Create a new data object for training
        training_set = self.data.clone()
        training_set.edge_index = training_edge_index

        return training_set




