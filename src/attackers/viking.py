"""VIKING — supervised network-poisoning attack (Gupta & Chakraborty, PAKDD 2021, arXiv:2102.07164).

Faithful port to our targeted, removal-only, label-free, surrogate-transfer rank-demotion setting.

VIKING scores each candidate edge by its closed-form rank-1 effect on a node's COMMUNITY-PURITY
objective (``midval = A @ L`` over a label matrix ``L``) — using only the adjacency and labels, with NO
node features, NO model weights, and NO gradients — then greedily removes the top-scored edges. That
structure-only, model-agnostic signal is what makes VIKING distinct from nettack (features + surrogate-
GCN classification margin) and agpoa (gradient embedding-drift).

We use the paper's semi-supervised variant (VIKING-s): the label matrix ``L`` is PSEUDO-labels clustered
(KMeans) from the SURROGATE's clean embeddings, which keeps the method label-free and transferable to
the victim. The objective stays VIKING's purity loss (NOT the centroid-drift loss the other ports use —
that would collapse VIKING into a clone). Removing a target's same-pseudo-class (homophilous) neighbors
lowers its neighborhood class purity, stripping the edges that anchor it near its community centroid —
the centroid the rank-demotion metric ranks against.

Original code: https://github.com/virresh/viking
(node_embedding_attack/perturbation_attack.py :: estimate_loss_with_supervision_type2).
Set ``use_true_labels: true`` for the fully-supervised VIKING (uses ``data.y``; this leaves the
label-free black-box threat model — off by default).
"""
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from src.attackers.base import Attacker
from src.attackers.surrogate_retriever import get_or_train_surrogate_retriever
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class VikingAttack(Attacker):
    """VIKING-s community-purity poisoning, ported to targeted edge-removal rank demotion."""

    def __init__(self, cfg, system_config: DictConfig, data: Data, retriever=None, **kwargs):
        if retriever is None:
            raise ValueError("VikingAttack requires a retriever (for the surrogate / pseudo-labels)")
        super().__init__(cfg, system_config, data)
        self.sort_reverse = True   # higher purity-drop = remove first

        self.num_pseudo_classes = int(getattr(cfg, "num_pseudo_classes", 7))
        self.use_true_labels = bool(getattr(cfg, "use_true_labels", False))

        # Surrogate (label-free, distilled from the surrogate retriever's clean embeddings) — the SAME
        # block as nettack/agpoa, so VIKING-s transfers black-box to the victim. The surrogate model
        # itself is unused for scoring; we only need its clean embeddings to derive pseudo-labels.
        surrogate, clean_embeddings = get_or_train_surrogate_retriever(
            data, retriever, system_config.dataset.name, system_config.retriever.name,
            graph_model=getattr(system_config.general, "graph_model", "gcn"),
            cache_dir=getattr(system_config.general, "cache_dir", "cache"),
            hidden_channels=getattr(cfg, "surrogate_hidden", 256),
            epochs=getattr(cfg, "surrogate_epochs", 200),
            lr=getattr(cfg, "surrogate_lr", 0.01),
            weight_decay=getattr(cfg, "surrogate_weight_decay", 5e-4),
            dropout=getattr(cfg, "surrogate_dropout", 0.0),
            loss_type=getattr(cfg, "surrogate_loss_type", "cosine"),
            patience=getattr(cfg, "surrogate_patience", 30),
            device=str(self.device), run_id=None, seed=self.seed,
        )
        self.retriever_surrogate_time = float(getattr(surrogate, "retriever_surrogate_time", 0.0))

        # Label matrix L [N, C], one-hot. VIKING-s: pseudo-labels from KMeans on surrogate clean embs.
        if self.use_true_labels:
            labels = data.y.cpu().numpy()
            C = int(labels.max()) + 1
        else:
            zc = clean_embeddings.detach().cpu().numpy()
            C = max(2, min(self.num_pseudo_classes, zc.shape[0]))
            labels = KMeans(n_clusters=C, random_state=self.seed, n_init=10).fit_predict(zc)
        n = data.num_nodes
        self._L = torch.zeros((n, C), dtype=torch.float32, device=self.device)
        self._L[torch.arange(n, device=self.device),
                torch.tensor(labels, dtype=torch.long, device=self.device)] = 1.0
        logger.info(f"VIKING initialised: {'true' if self.use_true_labels else 'pseudo'}-labels, "
                    f"C={C}, N={n}")

    def score_edges(self, target_node) -> dict:
        """Per-edge community-purity drop at the target (VIKING supervised loss, target-restricted).

        ``midval_t = (A @ L)[target]`` is the target's neighbor-label mass; ``mu = (midval_t·L_t)/Σmidval_t``
        is the fraction of the target's neighbor mass that lands on its OWN (pseudo)class. Removing edge
        ``(target, nb)`` subtracts ``L[nb]`` (rank-1 update); the score is the resulting drop in ``mu``.
        Positive => the removal lowers the target's class purity (a good demotion edge). ``sort_reverse``
        is True, so base ``attack()`` removes the largest-drop edges first.
        """
        if isinstance(target_node, torch.Tensor):
            target_node = int(target_node.item())
        edges = self.get_target_edges(target_node)
        if not edges:
            return {}
        L = self._L
        nbrs = [self.get_edge_neighbor(e, target_node) for e in edges]
        nbr_idx = torch.tensor(nbrs, dtype=torch.long, device=self.device)
        midval_t = L[nbr_idx].sum(0)            # [C] = (A @ L)[target] over incident neighbors
        Lt = L[target_node]                     # [C] target's own (pseudo)class one-hot
        den = float(midval_t.sum())
        mu_clean = float((midval_t * Lt).sum()) / den if den > 0 else 0.0
        scores = {}
        for e, nb in zip(edges, nbrs):
            new_row = midval_t - L[nb]          # rank-1 removal update of the target's row
            new_den = float(new_row.sum())
            mu_after = float((new_row * Lt).sum()) / new_den if new_den > 0 else 0.0
            scores[e] = mu_clean - mu_after     # purity DROP; >0 = removal hurts the target's class purity
        return scores
