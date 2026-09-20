"""
Part 8: Deep IDS baselines (9 models): MLP, 1D-CNN, LSTM, GRU, CNN-LSTM,
CNN-BiLSTM, TCN, Transformer-encoder, FT-Transformer(-style).

Sequence models (everything except MLP/FT-Transformer) consume a sliding window
of `seq_len` consecutive flows from the same leakage-safe split (row order
within a split), predicting the label of the *last* flow in the window -- the
same temporal-windowing idea used by the CNN-BiLSTM reference paper read for
style (Trans2), adapted here to flow-level tabular IDS data rather than
demand/price series. MLP and FT-Transformer instead consume a single flow's
feature vector (no temporal window), representing the "no temporal context"
deep-baseline family.

All models are intentionally small (2-3 layers, <100K params) since this is a
baseline suite meant for fair comparison against the equally lightweight RL
candidates, not a SOTA architecture search.
"""
import time, tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.metrics_utils import ids_metrics_from_scores


class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SeqFlowDataset(Dataset):
    """Sliding window of seq_len consecutive rows; label = last row's label."""
    def __init__(self, X, y, seq_len=16):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.n = len(X)

    def __len__(self):
        return max(self.n - self.seq_len + 1, 0)

    def __getitem__(self, idx):
        seq = self.X[idx: idx + self.seq_len]
        label = self.y[idx + self.seq_len - 1]
        return seq, label


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, in_dim, seq_len, hidden=32, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim, hidden, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                   nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: [B, T, F]
        h = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.fc(h)


class RNNBaseline(nn.Module):
    def __init__(self, in_dim, hidden=32, n_classes=2, cell="lstm", bidir=False):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(in_dim, hidden, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class CNNRNN(nn.Module):
    def __init__(self, in_dim, hidden=32, n_classes=2, bidir=False):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, hidden, 3, padding=1)
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        h = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        out, _ = self.rnn(h)
        return self.fc(out[:, -1, :])


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.shape[-1]]


class TCN(nn.Module):
    def __init__(self, in_dim, hidden=32, n_classes=2):
        super().__init__()
        self.b1 = TCNBlock(in_dim, hidden, dilation=1)
        self.b2 = TCNBlock(hidden, hidden, dilation=2)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = F.relu(self.b1(x.transpose(1, 2)))
        h = F.relu(self.b2(h))
        h = h.mean(dim=-1)
        return self.fc(h)


class TransformerEncoderBaseline(nn.Module):
    def __init__(self, in_dim, hidden=32, n_classes=2, nhead=2, nlayers=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=64,
                                            batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        return self.fc(h[:, -1, :])


class FTTransformer(nn.Module):
    """Per-feature tokenization (each scalar feature -> its own token) + a small
    transformer encoder + CLS-token readout; single-flow input (no temporal window)."""
    def __init__(self, in_dim, hidden=16, n_classes=2, nhead=2, nlayers=1):
        super().__init__()
        self.token_proj = nn.Linear(1, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=32,
                                            batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: [B, F]
        b, f = x.shape
        tok = self.token_proj(x.unsqueeze(-1))  # [B, F, hidden]
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = self.enc(h)
        return self.fc(h[:, 0, :])


def train_and_eval(model, train_ds, eval_ds, epochs=6, batch_size=256, lr=1e-3, class_weight=None,
                    seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False)
    scores, trues = [], []
    t1 = time.time()
    with torch.no_grad():
        for xb, yb in eval_loader:
            logits = model(xb)
            p = F.softmax(logits, dim=-1)[:, 1]
            scores.append(p.numpy())
            trues.append(yb.numpy())
    infer_time = (time.time() - t1) / max(len(eval_ds), 1)
    scores = np.concatenate(scores)
    trues = np.concatenate(trues)
    n_params = sum(p.numel() for p in model.parameters())
    return trues, scores, train_time, infer_time, n_params


def run_all(X_train, y_train, X_eval, y_eval, seed=42, seq_len=16, epochs=6, eval_split="val"):
    in_dim = X_train.shape[1]
    class_weight = torch.tensor([1.0, max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 0.2)],
                                 dtype=torch.float32)

    flat_train, flat_eval = FlowDataset(X_train, y_train), FlowDataset(X_eval, y_eval)
    seq_train = SeqFlowDataset(X_train, y_train, seq_len=seq_len)
    seq_eval = SeqFlowDataset(X_eval, y_eval, seq_len=seq_len)

    specs = [
        ("MLP", MLP(in_dim), flat_train, flat_eval),
        ("1D-CNN", CNN1D(in_dim, seq_len), seq_train, seq_eval),
        ("LSTM", RNNBaseline(in_dim, cell="lstm"), seq_train, seq_eval),
        ("GRU", RNNBaseline(in_dim, cell="gru"), seq_train, seq_eval),
        ("CNN-LSTM", CNNRNN(in_dim, bidir=False), seq_train, seq_eval),
        ("CNN-BiLSTM", CNNRNN(in_dim, bidir=True), seq_train, seq_eval),
        ("TCN", TCN(in_dim), seq_train, seq_eval),
        ("TransformerEncoder", TransformerEncoderBaseline(in_dim), seq_train, seq_eval),
        ("FT-Transformer", FTTransformer(in_dim), flat_train, flat_eval),
    ]

    results = []
    for name, model, tr_ds, ev_ds in specs:
        tracemalloc.start()
        trues, scores, train_time, infer_time, n_params = train_and_eval(
            model, tr_ds, ev_ds, epochs=epochs, class_weight=class_weight, seed=seed)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics = ids_metrics_from_scores(trues, scores)
        metrics.update({"model": name, "family": "deep", "train_time_sec": train_time,
                         "inference_latency_sec_per_sample": infer_time, "peak_memory_MB": peak / 1e6,
                         "n_params": n_params, "seed": seed, "eval_split": eval_split})
        results.append(metrics)
        print(f"  {name}: macro_f1={metrics['macro_f1']:.4f} minority_recall={metrics['minority_recall']:.4f} "
              f"pr_auc={metrics['pr_auc']:.4f} n_params={n_params}", flush=True)
    return results
