"""
EEGNet for EEG emotion decoding — LOSO evaluation on GAMEEMO.

Architecture (Lawhern et al. 2018):
  Block 1: Temporal conv (F1=8, kern=64) → DepthwiseConv (D=2) → BN → ELU → AvgPool → Dropout
  Block 2: SeparableConv (F2=16, kern=16) → BN → ELU → AvgPool → Dropout
  Classifier: Flatten → Linear(n_classes)

Input: (batch, 1, C, T) = (batch, 1, 14, 256)

Run:
  conda run -n bci python src/models/eegnet.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED  = 42
EPOCHS       = 150
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20       # early-stop patience (epochs without val-loss improvement)
VAL_FRAC     = 0.15     # fraction of training subjects held out for early stopping

# EEGNet hyper-params
F1 = 8    # temporal filters
D  = 2    # depth multiplier → F2 = F1*D = 16 spatial filters
F2 = F1 * D
KERN_TEMP = 64    # temporal kernel ≈ 0.5 s at 128 Hz
KERN_SEP  = 16    # separable kernel ≈ 0.125 s
DROP_RATE = 0.5

N_CHANNELS = 14
N_SAMPLES  = 256
N_CLASSES  = 4

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, C=N_CHANNELS, T=N_SAMPLES,
                 F1=F1, D=D, F2=F2,
                 kern_temp=KERN_TEMP, kern_sep=KERN_SEP, drop_rate=DROP_RATE):
        super().__init__()

        # Block 1 — temporal + depthwise spatial
        self.block1 = nn.Sequential(
            # Temporal conv: learn frequency-band filters
            nn.Conv2d(1, F1, (1, kern_temp), padding=(0, kern_temp // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise: learn spatial (channel) filters per temporal filter
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_rate),
        )

        # Block 2 — separable conv: depthwise + pointwise
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, kern_sep), padding=(0, kern_sep // 2),
                      groups=F2, bias=False),          # depthwise
            nn.Conv2d(F2, F2, (1, 1), bias=False),    # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_rate),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            out = self.block2(self.block1(dummy))
            flat = out.numel()

        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X).unsqueeze(1),   # (N, 1, C, T)
        torch.from_numpy(y).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def _normalize(X_train, X_test):
    """Per-channel z-score using training statistics."""
    # X shape: (N, C, T)
    N, C, T = X_train.shape
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, C, T)
    N2 = X_test.shape[0]
    X_te = scaler.transform(X_test.reshape(N2, -1)).reshape(N2, C, T)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def train_subject_fold(X_train, y_train, X_test, y_test, verbose=False):
    """Train EEGNet on one LOSO fold, return (y_pred, y_prob)."""
    torch.manual_seed(RANDOM_SEED)

    X_train, X_test = _normalize(X_train, X_test)

    # Split a small validation set from training subjects for early stopping
    n_train = len(y_train)
    n_val   = max(1, int(n_train * VAL_FRAC))
    rng     = np.random.default_rng(RANDOM_SEED)
    perm    = rng.permutation(n_train)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    train_loader = _make_loader(X_train[tr_idx],  y_train[tr_idx],  BATCH_SIZE, shuffle=True)
    val_loader   = _make_loader(X_train[val_idx], y_train[val_idx], BATCH_SIZE, shuffle=False)
    test_loader  = _make_loader(X_test,           y_test,           BATCH_SIZE, shuffle=False)

    model     = EEGNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(yb)
        val_loss /= n_val

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                if verbose:
                    print(f"      early stop @ epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(DEVICE))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_probs.append(torch.softmax(logits, 1).cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def evaluate_loso(X, y, subject_ids, verbose=True):
    subjects = np.unique(subject_ids)
    records  = []

    for subj in subjects:
        test_mask  = subject_ids == subj
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        y_pred, y_prob = train_subject_fold(X_train, y_train, X_test, y_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

        records.append({"subject": int(subj), "acc": acc, "f1": f1, "auc": auc})
        if verbose:
            print(f"    S{subj:02d}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    accs = np.array([r["acc"] for r in records])
    f1s  = np.array([r["f1"]  for r in records])
    aucs = np.array([r["auc"] for r in records])

    return {
        "per_subject": records,
        "mean_acc": accs.mean(), "std_acc": accs.std(),
        "mean_f1":  f1s.mean(),  "std_f1":  f1s.std(),
        "mean_auc": aucs.mean(), "std_auc": aucs.std(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.loader import load_all

    print(f"Device: {DEVICE}")

    print("Loading data …")
    data = load_all(verbose=False)
    X    = data["X"]            # (33264, 14, 256)
    y    = data["y"]
    sids = data["subject_ids"]
    print(f"  X={X.shape}  y={y.shape}\n")

    print("=" * 55)
    print("  EEGNet  LOSO evaluation")
    print("=" * 55)
    t0  = time.time()
    res = evaluate_loso(X, y, sids, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  → mean acc={res['mean_acc']:.3f}±{res['std_acc']:.3f}  "
          f"f1={res['mean_f1']:.3f}±{res['std_f1']:.3f}  "
          f"auc={res['mean_auc']:.3f}±{res['std_auc']:.3f}  "
          f"({elapsed:.0f}s)")

    METRICS = Path(__file__).resolve().parents[2] / "results" / "metrics"
    rows = [{
        "model":    "EEGNet",
        "features": "raw",
        "acc_mean": round(res["mean_acc"], 4),
        "acc_std":  round(res["std_acc"],  4),
        "f1_mean":  round(res["mean_f1"],  4),
        "f1_std":   round(res["std_f1"],   4),
        "auc_mean": round(res["mean_auc"], 4),
        "auc_std":  round(res["std_auc"],  4),
    }]
    out = METRICS / "eegnet_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved → {out}")
