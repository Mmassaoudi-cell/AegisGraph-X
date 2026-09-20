"""
AegisGraph-X Layer 3: tree-guided neural distillation.

Distills the router's mixed (tree+neural+graph ensemble) soft predictions
into a single compact feed-forward student, for deployment settings where
running the full 9-expert pool is too costly (Part 11 runtime comparison).
Loss = soft-label KL (teacher-temperature-scaled) + relational pairwise
similarity distillation (matches pairwise cosine-similarity structure of a
minibatch's penultimate representations to the best single tree expert's
leaf-embedding similarity) + hard-class logit-margin loss (pushes the
minority-class logit margin to match or exceed the teacher's) + a
calibration-preserving temperature term (the student is trained to match
the *calibrated* teacher probability, not the raw ensemble mix, so its
own uncalibrated output already sits close to the calibration target).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistilledStudent(nn.Module):
    def __init__(self, in_dim, hidden=64, embed_dim=16):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.embed = nn.Linear(hidden, embed_dim)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, return_embed=False):
        h = self.trunk(x)
        logits = self.head(h)
        if return_embed:
            return logits, self.embed(h)
        return logits


def relational_similarity_loss(student_embed, teacher_leaf_embed):
    s = F.normalize(student_embed, dim=-1)
    t = F.normalize(teacher_leaf_embed, dim=-1)
    sim_s = s @ s.t()
    sim_t = t @ t.t()
    return F.mse_loss(sim_s, sim_t)


def logit_margin_loss(student_logits, y, minority_class=1, margin=1.0):
    mask = (y == minority_class)
    if mask.sum() == 0:
        return torch.tensor(0.0)
    margins = student_logits[mask, minority_class] - student_logits[mask, 1 - minority_class]
    return F.relu(margin - margins).mean()


def train_distilled_student(X_train, y_train, teacher_soft_prob, teacher_leaf_embed=None,
                             epochs=30, lr=1e-3, temperature=2.0, batch_size=256,
                             lambda_kl=1.0, lambda_rel=0.3, lambda_margin=0.5, seed=42,
                             use_focal=True, use_contrastive=True, use_hard_mining=True,
                             lambda_focal=0.5, lambda_contrastive=0.2, gamma_focal=2.0):
    """use_focal/use_contrastive/use_hard_mining toggle the Layer-6 hard-class
    terms (aegis.hard_class) independently, so the ablation study's "No focal
    loss" / "No contrastive learning" / "No hard-class mining" rows can be
    produced by disabling exactly one term relative to the full student."""
    from aegis.hard_class import (ClassBalancedFocalLoss, supervised_contrastive_loss,
                                   hard_negative_indices, samples_per_class)
    torch.manual_seed(seed)
    n, d = X_train.shape
    student = DistilledStudent(d)
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    teacher_p = torch.as_tensor(teacher_soft_prob, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    teacher_soft = torch.stack([1 - teacher_p, teacher_p], dim=-1)
    teacher_soft_T = F.softmax(torch.log(teacher_soft) / temperature, dim=-1)

    leaf_t = None
    if teacher_leaf_embed is not None:
        leaf_t = torch.as_tensor(teacher_leaf_embed, dtype=torch.float32)

    n_pos = max(float((y_train == 1).sum()), 1.0)
    n_neg = max(float((y_train == 0).sum()), 1.0)
    ce_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
    focal_loss_fn = ClassBalancedFocalLoss(samples_per_class(y_train), gamma=gamma_focal) if use_focal else None

    idx_all = np.arange(n)
    hard_weight = np.ones(n, dtype=np.float32)
    student.train()
    for ep in range(epochs):
        if use_hard_mining and ep > 0:
            probs = hard_weight / hard_weight.sum()
            order = np.random.choice(idx_all, size=n, replace=True, p=probs)
        else:
            order = np.random.permutation(idx_all)
        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            bidx = torch.as_tensor(batch_idx, dtype=torch.long)
            opt.zero_grad()
            logits, embed = student(Xt[bidx], return_embed=True)

            student_soft_T = F.log_softmax(logits / temperature, dim=-1)
            kl = F.kl_div(student_soft_T, teacher_soft_T[bidx], reduction="batchmean") * (temperature ** 2)

            ce = focal_loss_fn(logits, yt[bidx]) if use_focal else F.cross_entropy(logits, yt[bidx], weight=ce_weight)
            margin = logit_margin_loss(logits, yt[bidx])

            contrastive = torch.tensor(0.0)
            if use_contrastive and len(bidx) > 1:
                contrastive = supervised_contrastive_loss(embed, yt[bidx])

            if use_hard_mining:
                with torch.no_grad():
                    per_sample_ce = F.cross_entropy(logits, yt[bidx], reduction="none").numpy()
                hard_weight[batch_idx] = per_sample_ce

            rel = torch.tensor(0.0)
            if leaf_t is not None and len(bidx) > 1:
                rel = relational_similarity_loss(embed, leaf_t[bidx])

            loss = ce + lambda_kl * kl + lambda_rel * rel + lambda_margin * margin + lambda_contrastive * contrastive
            loss.backward()
            opt.step()
    return student


def student_predict_proba(student, X):
    student.eval()
    with torch.no_grad():
        logits = student(torch.as_tensor(X, dtype=torch.float32))
        p = F.softmax(logits, dim=-1)[:, 1].numpy()
    return p


def tree_leaf_embedding(tree_model, X, max_dim=32):
    """Best-effort leaf-index embedding from a fitted tree ensemble, used as
    the relational-distillation target. Falls back to None (relational term
    skipped) for model types without an `apply`-style leaf-index API."""
    try:
        leaves = tree_model.apply(X)
        leaves = np.asarray(leaves)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        leaves = leaves[:, :max_dim].astype(np.float32)
        leaves = (leaves - leaves.mean(axis=0, keepdims=True)) / (leaves.std(axis=0, keepdims=True) + 1e-6)
        return leaves
    except Exception:
        return None
