"""
Surrogate GCN model for Nettack attack.
Trains a simple 2-layer GCN that can be used to extract weights for Nettack scoring.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
import os


class SurrogateGCN(torch.nn.Module):
    """
    Simple 2-layer GCN for use as a surrogate model in Nettack.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_surrogate_gcn(data, hidden_channels=256, epochs=200, lr=0.01, weight_decay=5e-4, device='cuda'):
    """
    Train a surrogate GCN model on the given data.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data with train/val/test masks
    hidden_channels : int
        Number of hidden units
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    device : str
        Device to train on
        
    Returns
    -------
    SurrogateGCN
        Trained model
    """
    data = data.to(device)
    num_classes = int(data.y.max().item()) + 1
    
    model = SurrogateGCN(data.num_features, hidden_channels, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Check if we have train/val/test masks
    if not hasattr(data, 'train_mask'):
        # Create simple split if masks don't exist
        num_nodes = data.num_nodes
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        indices = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size+val_size]] = True
        data.test_mask[indices[train_size+val_size:]] = True
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def get_or_train_surrogate_gcn(data, data_name, cache_dir='output/surrogate_gcn', device='cuda'):
    """
    Get a cached surrogate GCN or train a new one.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data
    data_name : str
        Dataset name for caching
    cache_dir : str
        Directory to cache trained models
    device : str
        Device to use
        
    Returns
    -------
    SurrogateGCN
        Trained surrogate GCN model
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{data_name}_surrogate_gcn.pt')
    
    num_classes = int(data.y.max().item()) + 1
    
    # Try to load cached model
    if os.path.exists(cache_path):
        try:
            model = SurrogateGCN(data.num_features, 256, num_classes).to(device)
            model.load_state_dict(torch.load(cache_path, weights_only=True))
            model.eval()
            print(f"Loaded cached surrogate GCN from {cache_path}")
            return model
        except Exception as e:
            print(f"Failed to load cached model: {e}. Training new one...")
    
    # Train new model
    print(f"Training surrogate GCN for {data_name}...")
    model = train_surrogate_gcn(data, device=device)
    
    # Cache the model
    torch.save(model.state_dict(), cache_path)
    print(f"Saved surrogate GCN to {cache_path}")
    
    return model
