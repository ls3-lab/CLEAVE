"""Retriever-embedding surrogate GCN.

This module mirrors the surrogate idea used in GraphMI: query the victim
retriever once on the clean graph, then train a lightweight differentiable GCN
to reproduce those embeddings. Gradient-based attacks can then optimize through
the surrogate without backpropagating through the retriever itself.
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class SurrogateRetrieverGCN(nn.Module):
    """2-layer GCN that outputs retriever-sized embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.W1 = nn.Linear(in_channels, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = float(dropout)

    @staticmethod
    def symmetric_normalize(edge_index: torch.Tensor, edge_values: torch.Tensor, num_nodes: int) -> torch.Tensor:
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=edge_values.device, dtype=edge_values.dtype)
        deg.index_add_(0, row, edge_values)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = torch.where(torch.isinf(deg_inv_sqrt), torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        norm_values = deg_inv_sqrt[row] * edge_values * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(edge_index, norm_values, (num_nodes, num_nodes)).coalesce()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_values: torch.Tensor = None) -> torch.Tensor:
        num_nodes = x.size(0)
        if edge_index.dim() == 2 and edge_index.size(0) == num_nodes and edge_index.size(1) == num_nodes:
            h = edge_index @ self.W1(x)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            return edge_index @ self.W2(h)
        if edge_values is None:
            edge_values = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        adj_norm = self.symmetric_normalize(edge_index, edge_values, num_nodes)
        h = torch.sparse.mm(adj_norm, self.W1(x))
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return torch.sparse.mm(adj_norm, self.W2(h))


def _edge_index_with_self_loops(data, device):
    edge_index = data.edge_index.to(device)
    self_idx = torch.arange(data.num_nodes, device=device)
    self_loops = torch.stack([self_idx, self_idx])
    return torch.cat([edge_index, self_loops], dim=1)


def train_surrogate_retriever(
    data,
    retriever,
    *,
    hidden_channels=256,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    dropout=0.0,
    loss_type="cosine",
    patience=30,
    device="cuda",
    seed=None,
):
    """Train a surrogate GCN to reproduce retriever embeddings."""
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    data = data.to(device)
    with torch.no_grad():
        target_embeddings = retriever.encode(data.x, data.edge_index).detach()

    model = SurrogateRetrieverGCN(
        data.num_features,
        int(hidden_channels),
        target_embeddings.size(1),
        dropout=float(dropout),
    ).to(device)
    optimizer = Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    edge_index = _edge_index_with_self_loops(data, device)

    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, edge_index)
        if loss_type == "cosine":
            loss = 1.0 - F.cosine_similarity(pred, target_embeddings, dim=1).mean()
        else:
            loss = F.mse_loss(pred, target_embeddings)
        loss.backward()
        optimizer.step()

        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, edge_index)
        cos = F.cosine_similarity(pred, target_embeddings, dim=1).mean().item()
        mse = F.mse_loss(pred, target_embeddings).item()
    logger.info(
        f"Retriever surrogate trained: loss={best_loss:.6f}, mse={mse:.6f}, "
        f"avg_cos_sim={cos:.4f}, epochs={epoch + 1}"
    )
    return model, target_embeddings


def get_or_train_surrogate_retriever(
    data,
    retriever,
    data_name,
    retriever_name,
    *,
    graph_model="gcn",
    cache_dir="cache",
    hidden_channels=256,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    dropout=0.0,
    loss_type="cosine",
    patience=30,
    device="cuda",
    run_id=None,
    seed=None,
):
    """Load a cached retriever surrogate or train a new one."""
    timer_start = time.perf_counter()
    device = str(device)
    ckpt_dir = os.path.join(cache_dir, "checkpoints", retriever_name, data_name, "surrogate_retriever")
    os.makedirs(ckpt_dir, exist_ok=True)

    with torch.no_grad():
        target_embeddings = retriever.encode(data.x.to(device), data.edge_index.to(device)).detach()

    out_dim = int(target_embeddings.size(1))
    run_suffix = f"_{run_id}" if run_id is not None else ""
    # Tag the cache with the SOURCE checkpoint that produced target_embeddings (retriever.encode above).
    # A surrogate distilled from the VICTIM must never be served to an attacker that (correctly, post-H1)
    # distills the SURROGATE checkpoint, or vice versa. The old key omitted this, so stale victim-distilled
    # caches silently collapsed the black-box transfer gap (H1). Encoding the source in the filename makes a
    # wrong-source cache a clean MISS (-> retrain from the right source), and never a wrong-source hit.
    try:
        src_tag = os.path.splitext(os.path.basename(retriever.get_model_path()))[0]
    except Exception:
        src_tag = "oracle" if getattr(retriever, "use_oracle", False) else "victim"
    cache_path = os.path.join(
        ckpt_dir,
        f"best_{graph_model}_hidden{int(hidden_channels)}_out{out_dim}_{loss_type}_src-{src_tag}{run_suffix}.pt",
    )

    if os.path.exists(cache_path):
        try:
            ckpt = torch.load(cache_path, map_location=device)
            model = SurrogateRetrieverGCN(
                ckpt["in_channels"],
                ckpt["hidden_channels"],
                ckpt["out_channels"],
                dropout=ckpt.get("dropout", dropout),
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            elapsed = time.perf_counter() - timer_start
            model.retriever_surrogate_time = elapsed
            model.retriever_surrogate_cache_path = cache_path
            model.retriever_surrogate_loaded_from_cache = True
            logger.info(f"Loaded retriever surrogate from {cache_path} in {elapsed:.4f}s")
            return model, target_embeddings
        except Exception as exc:
            logger.warning(f"Failed to load retriever surrogate from {cache_path}: {exc}. Training new one.")

    model, target_embeddings = train_surrogate_retriever(
        data,
        retriever,
        hidden_channels=hidden_channels,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        loss_type=loss_type,
        patience=patience,
        device=device,
        seed=seed,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": data.num_features,
            "hidden_channels": int(hidden_channels),
            "out_channels": out_dim,
            "dropout": float(dropout),
            "loss_type": loss_type,
            "run_id": run_id,
            "seed": seed,
            "source": src_tag,
        },
        cache_path,
    )
    elapsed = time.perf_counter() - timer_start
    model.retriever_surrogate_time = elapsed
    model.retriever_surrogate_cache_path = cache_path
    model.retriever_surrogate_loaded_from_cache = False
    logger.info(f"Saved retriever surrogate to {cache_path} (elapsed={elapsed:.4f}s)")
    return model, target_embeddings
