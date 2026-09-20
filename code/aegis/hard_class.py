"""
AegisGraph-X Layer 6: hard-class / minority-class learning utilities.

Class-balanced focal loss, supervised contrastive loss, hard-negative
mining (topk highest-loss majority-class samples per batch), and rare-class
threshold calibration (delegates to calibrate.select_threshold_for_fpr_budget
/ select_threshold_max_f1 computed per-class on validation data only).
These are used while training the distilled student (distill.py) and,
optionally, the neural experts (expert_pool.py) — toggled off one at a time
in the ablation study (Part 8: "No contrastive learning", "No focal loss",
"No hard-class mining").
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """Cui et al. (2019)-style class-balanced term (effective number of
    samples, beta close to 1) combined with a focal modulating factor."""
    def __init__(self, samples_per_class, gamma=2.0, beta=0.999):
        super().__init__()
        eff_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer_weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, logits, y):
        w = self.register_buffer_weights.to(logits.device)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        logp_t = logp.gather(1, y.view(-1, 1)).squeeze(1)
        p_t = p.gather(1, y.view(-1, 1)).squeeze(1)
        w_t = w[y]
        loss = -w_t * ((1 - p_t) ** self.gamma) * logp_t
        return loss.mean()


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    z = F.normalize(embeddings, dim=-1)
    sim = (z @ z.t()) / temperature
    n = z.shape[0]
    mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask_self, -1e9)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & (~mask_self)

    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    pos_counts = pos_mask.sum(dim=-1).clamp_min(1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=-1) / pos_counts
    valid = pos_mask.sum(dim=-1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    return -mean_log_prob_pos[valid].mean()


def hard_negative_indices(losses, y, majority_class=0, top_frac=0.3):
    """Given per-sample losses and labels, return indices of the
    highest-loss majority-class ("easy" class, usually benign) samples --
    the hardest majority examples to correctly reject -- for reweighting or
    oversampling in the next epoch."""
    maj_idx = np.where(y == majority_class)[0]
    if len(maj_idx) == 0:
        return np.array([], dtype=int)
    maj_losses = losses[maj_idx]
    k = max(1, int(top_frac * len(maj_idx)))
    order = np.argsort(-maj_losses)[:k]
    return maj_idx[order]


def samples_per_class(y, n_classes=2):
    return np.array([np.sum(y == c) for c in range(n_classes)], dtype=np.float64)
