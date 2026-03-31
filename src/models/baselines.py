"""
Baseline classifiers for EEG emotion decoding.

Model:
  - XGBoost : gradient-boosted trees, 200 estimators, tuned for multiclass

Evaluation:
  - LOSO (Leave-One-Subject-Out) cross-validation via evaluate_within_subject()
  - Metrics: accuracy, macro F1, one-vs-rest AUC-ROC
"""

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_xgboost() -> XGBClassifier:
    """XGBoost multiclass with fixed hyperparameters."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def evaluate_within_subject(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Leave-One-Subject-Out evaluation.

    For each unique subject S:
      - Train on all epochs from every other subject
      - Predict on all epochs from S
      - Record accuracy, macro F1, one-vs-rest AUC-ROC

    Parameters
    ----------
    X           : (n_epochs, n_features)
    y           : (n_epochs,)  integer class labels
    subject_ids : (n_epochs,)  integer subject IDs
    verbose     : print per-subject results

    Returns
    -------
    dict with keys: per_subject, mean_acc, std_acc, mean_f1, std_f1, mean_auc, std_auc
    """
    subjects = np.unique(subject_ids)
    records = []

    for subj in subjects:
        test_mask  = subject_ids == subj
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        model = make_xgboost()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

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

    METRICS = Path(__file__).resolve().parents[2] / "results" / "metrics"

    print("Loading features …")
    X_bp = np.load(METRICS / "features_bandpower.npy")
    X_de = np.load(METRICS / "features_de.npy")
    y    = np.load(METRICS / "labels.npy")
    sids = np.load(METRICS / "subject_ids.npy")
    print(f"  Band power : {X_bp.shape}   DE : {X_de.shape}   labels : {y.shape}\n")

    conditions = [
        ("XGBoost", "bandpower", X_bp),
        ("XGBoost", "DE",        X_de),
    ]

    rows = []
    for model_name, feat_name, X in conditions:
        print(f"{'='*55}")
        print(f"  {model_name} + {feat_name}")
        print(f"{'='*55}")
        t0 = time.time()
        res = evaluate_within_subject(X, y, sids, verbose=True)
        elapsed = time.time() - t0
        print(f"  → mean acc={res['mean_acc']:.3f}±{res['std_acc']:.3f}  "
              f"f1={res['mean_f1']:.3f}±{res['std_f1']:.3f}  "
              f"auc={res['mean_auc']:.3f}±{res['std_auc']:.3f}  "
              f"({elapsed:.0f}s)\n")
        rows.append({
            "model":    model_name,
            "features": feat_name,
            "acc_mean": round(res["mean_acc"], 4),
            "acc_std":  round(res["std_acc"],  4),
            "f1_mean":  round(res["mean_f1"],  4),
            "f1_std":   round(res["std_f1"],   4),
            "auc_mean": round(res["mean_auc"], 4),
            "auc_std":  round(res["std_auc"],  4),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 75)
    print("RESULTS TABLE — LOSO cross-validation (28 subjects, 4-class)")
    print("=" * 75)
    print(f"{'Model':<10} {'Features':<12} {'Accuracy':>12} {'Macro F1':>12} {'AUC-ROC':>12}")
    print("-" * 75)
    for _, r in df.iterrows():
        print(f"{r['model']:<10} {r['features']:<12} "
              f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}   "
              f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}   "
              f"{r['auc_mean']:.3f}±{r['auc_std']:.3f}")
    print("=" * 75)

    out_path = METRICS / "baseline_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
