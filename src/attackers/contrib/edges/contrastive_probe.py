"""THP-guided contrastive probe encoder (GRACE-style, EPAGCL-inspired).

Trains GNN encoder(s) with NT-Xent contrastive learning where the edge-drop
augmentation is GUIDED BY THP: low-THP (unimportant) edges are dropped with
higher probability, so the encoder learns representations invariant to
unimportant-edge removal and SENSITIVE to important-edge removal. The resulting
embedding is a learned "important-edge-aware" view of the graph, used as extra
node features for the EDGES v2 ranker. Training multiple encoders (n_views)
gives an ensemble multi-view representation.

Access-free: uses only graph structure + raw features + THP (no oracle, no
victim/surrogate retriever).
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

logger = logging.getLogger(__name__)


class _Encoder(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, out)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        return self.c2(x, edge_index)


class _Proj(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)

    def forward(self, z):
        return self.l2(F.elu(self.l1(z)))


def _nt_xent(z1, z2, tau):
    """GRACE InfoNCE on a batch of node pairs (z1[i], z2[i] are positives)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    def _dir(a, b):
        ab = torch.exp((a @ b.t()) / tau)
        aa = torch.exp((a @ a.t()) / tau)
        pos = torch.diag(ab)
        denom = ab.sum(1) + aa.sum(1) - torch.diag(aa)
        return -torch.log(pos / denom).mean()

    return 0.5 * (_dir(z1, z2) + _dir(z2, z1))


def _thp_keep_prob(edge_index, thp_scores, num_edges):
    """Per-edge keep-probability prior in [0,1] from THP (rank-normalized).
    Higher THP -> higher keep prob (important edges preserved in augmentation)."""
    vals = torch.tensor(
        [thp_scores.get((edge_index[0, i].item(), edge_index[1, i].item()), 0.0)
         for i in range(num_edges)], dtype=torch.float)
    # rank-normalize to [0,1] (robust to THP's scale/sign)
    order = vals.argsort().argsort().float()
    return order / max(1, num_edges - 1)


def _augment(x, edge_index, keep_prior, drop_rate, feat_mask_rate, gen, device):
    """One augmented view: THP-guided edge drop + random feature masking."""
    E = edge_index.size(1)
    # keep prob: important edges (keep_prior~1) almost always kept; unimportant dropped up to drop_rate
    keep_prob = 1.0 - drop_rate * (1.0 - keep_prior)
    keep = torch.rand(E, generator=gen, device=device) < keep_prob
    ei = edge_index[:, keep]
    # feature mask
    mask = (torch.rand(x.size(1), generator=gen, device=device) >= feat_mask_rate).float()
    return x * mask.unsqueeze(0), ei


def train_thp_contrastive_embedding(
        data, thp_scores, *, hid=128, out=64, n_views=2, epochs=300,
        drop_rate=0.4, feat_mask_rate=0.2, tau=0.5, batch=2048, lr=1e-3,
        device='cuda', seed=42):
    """Train n_views THP-guided contrastive encoders; return concatenated,
    L2-normalized node embeddings [N, out*n_views] (CPU tensor)."""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    x = data.x.float().to(dev)
    edge_index = data.edge_index.to(dev)
    N = x.size(0)
    keep_prior = _thp_keep_prob(data.edge_index, thp_scores, data.edge_index.size(1)).to(dev)

    views = []
    for v in range(n_views):
        gen = torch.Generator(device=dev).manual_seed(seed + 1000 * v)
        torch.manual_seed(seed + 1000 * v)
        enc = _Encoder(x.size(1), hid, out).to(dev)
        proj = _Proj(out).to(dev)
        opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=lr)
        enc.train(); proj.train()
        for ep in range(1, epochs + 1):
            x1, ei1 = _augment(x, edge_index, keep_prior, drop_rate, feat_mask_rate, gen, dev)
            x2, ei2 = _augment(x, edge_index, keep_prior, drop_rate, feat_mask_rate, gen, dev)
            z1 = proj(enc(x1, ei1))
            z2 = proj(enc(x2, ei2))
            idx = torch.randperm(N, generator=gen, device=dev)[:batch]
            loss = _nt_xent(z1[idx], z2[idx], tau)
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 100 == 0:
                logger.info(f"[cprobe] view {v} epoch {ep} | nt_xent={loss.item():.4f}")
        enc.eval()
        with torch.no_grad():
            emb = F.normalize(enc(x, edge_index), dim=1)
        views.append(emb.cpu())
    out_emb = torch.cat(views, dim=1)
    logger.info(f"[cprobe] trained {n_views} encoders -> embedding {tuple(out_emb.shape)}")
    return out_emb
