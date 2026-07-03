"""From-scratch SELF-SUPERVISED contrastive surrogate (GRACE-style).

Trains a 2-layer GCN (the SAME SurrogateRetrieverGCN the gradient attack backprops
through) with NT-Xent graph contrastive learning on PUBLIC graph structure + features
ONLY. It NEVER queries the victim retriever (no retriever.encode, no victim checkpoint)
— unlike get_or_train_surrogate_retriever, which distills the victim's embeddings. The
result is a black-box, victim-independent encoder whose geometry (same self-supervised
contrastive family as the GRACE/GCA/EPAGCL victims) is a transfer proxy for demotion.

User-confirmed black-box: a from-scratch SSL encoder using only structure+features is an
allowed black-box method (it touches no victim embeddings/gradients).
"""
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.attackers.surrogate_retriever import SurrogateRetrieverGCN
from src.attackers.contrib.edges.contrastive_probe import _nt_xent, _Proj

logger = logging.getLogger(__name__)


def _self_loops(edge_index, n, dev):
    si = torch.arange(n, device=dev)
    return torch.cat([edge_index, torch.stack([si, si])], dim=1)


def _augment(x, edge_index, drop_rate, feat_mask_rate, gen, dev):
    """One GRACE view: random edge drop + random feature masking."""
    E = edge_index.size(1)
    keep = torch.rand(E, generator=gen, device=dev) >= drop_rate
    ei = edge_index[:, keep]
    mask = (torch.rand(x.size(1), generator=gen, device=dev) >= feat_mask_rate).float()
    return x * mask.unsqueeze(0), ei


def train_contrastive_surrogate(data, *, hidden=256, out=256, epochs=300, lr=1e-3,
                                drop1=0.3, drop2=0.4, fmask1=0.2, fmask2=0.3,
                                tau=0.5, batch=2048, device='cuda', seed=42):
    """Train and return a SurrogateRetrieverGCN via GRACE NT-Xent (no victim access)."""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    x = data.x.float().to(dev)
    ei = data.edge_index.to(dev)
    N = x.size(0)
    model = SurrogateRetrieverGCN(x.size(1), int(hidden), int(out), dropout=0.0).to(dev)
    proj = _Proj(int(out)).to(dev)
    opt = Adam(list(model.parameters()) + list(proj.parameters()), lr=float(lr))
    gen = torch.Generator(device=dev).manual_seed(seed)
    model.train(); proj.train()
    t0 = time.time()
    for ep in range(1, int(epochs) + 1):
        x1, ei1 = _augment(x, ei, drop1, fmask1, gen, dev)
        x2, ei2 = _augment(x, ei, drop2, fmask2, gen, dev)
        z1 = proj(model(x1, _self_loops(ei1, N, dev)))
        z2 = proj(model(x2, _self_loops(ei2, N, dev)))
        idx = torch.randperm(N, generator=gen, device=dev)[:batch]
        loss = _nt_xent(z1[idx], z2[idx], tau)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == 1:
            logger.info(f"[cleave-surrogate] epoch {ep}/{epochs} | nt_xent={loss.item():.4f}")
    model.eval()
    logger.info(f"[cleave-surrogate] trained in {time.time()-t0:.1f}s (N={N}, hidden={hidden}, out={out})")
    return model


def get_or_train_contrastive_surrogate(data, data_name, *, cache_dir='cache', hidden=256,
                                       out=256, epochs=300, lr=1e-3, drop1=0.3, drop2=0.4,
                                       fmask1=0.2, fmask2=0.3, tau=0.5, batch=2048,
                                       device='cuda', seed=42, cache_tag=''):
    """Load cached SSL surrogate (retriever-INDEPENDENT: keyed by dataset only) or train.
    cache_tag = graph-fraction tag (e.g. '_g50'). Non-default drop/fmask also append to the key so
    augmentation sweeps (S6) train fresh surrogates; default values keep the original filename
    (backward-compatible with all existing caches)."""
    dev = str(device)
    ck_dir = os.path.join(cache_dir, 'checkpoints', 'cleave_surrogate', data_name)
    os.makedirs(ck_dir, exist_ok=True)
    tag = cache_tag
    if (round(drop1, 4), round(drop2, 4)) != (0.3, 0.4):
        tag += f'_d{drop1}-{drop2}'
    if (round(fmask1, 4), round(fmask2, 4)) != (0.2, 0.3):
        tag += f'_f{fmask1}-{fmask2}'
    path = os.path.join(ck_dir, f'grace_h{int(hidden)}_o{int(out)}_e{int(epochs)}{tag}_s{int(seed)}.pt')
    if os.path.exists(path):
        try:
            ck = torch.load(path, map_location=dev)
            model = SurrogateRetrieverGCN(ck['in_channels'], ck['hidden'], ck['out'], dropout=0.0).to(dev)
            model.load_state_dict(ck['state_dict']); model.eval()
            logger.info(f"[cleave-surrogate] loaded {path}")
            return model
        except Exception as e:
            logger.warning(f"[cleave-surrogate] load failed ({e}); retraining")
    model = train_contrastive_surrogate(
        data, hidden=hidden, out=out, epochs=epochs, lr=lr, drop1=drop1, drop2=drop2,
        fmask1=fmask1, fmask2=fmask2, tau=tau, batch=batch, device=device, seed=seed)
    torch.save({'state_dict': model.state_dict(), 'in_channels': data.num_features,
                'hidden': int(hidden), 'out': int(out), 'seed': int(seed)}, path)
    logger.info(f"[cleave-surrogate] saved {path}")
    return model
