"""
Data loading pipeline for the GAMEEMO dataset.

GAMEEMO: 28 subjects, 4 games (emotions), 14-channel EEG at 128 Hz.
Labels: G1=boring(0), G2=calm(1), G3=horror(2), G4=funny(3)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — single source of truth so downstream code never hardcodes these
# ---------------------------------------------------------------------------
CONFIG = {
    "SFREQ": 128,          # Hz — Emotiv EPOC sampling rate
    "WINDOW_SEC": 2,       # seconds per epoch — long enough to capture theta/alpha cycles
    "OVERLAP": 0.5,        # 50% overlap — standard for EEG to avoid losing boundary events
    "N_CHANNELS": 14,      # AF3 AF4 F3 F4 F7 F8 FC5 FC6 O1 O2 P7 P8 T7 T8
    "N_CLASSES": 4,        # boring calm horror funny
    "RANDOM_SEED": 42,
}

WINDOW_SAMPLES = CONFIG["SFREQ"] * CONFIG["WINDOW_SEC"]          # 256
STEP_SAMPLES   = int(WINDOW_SAMPLES * (1 - CONFIG["OVERLAP"]))   # 128

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "GAMEEMO"
# Actual extracted path has a long folder name; find it robustly
_candidates = list(DATA_ROOT.glob("*/GAMEEMO"))
if _candidates:
    GAMEEMO_ROOT = _candidates[0]
else:
    GAMEEMO_ROOT = DATA_ROOT / "GAMEEMO"

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}   # game number → class index
LABEL_NAMES = {0: "boring", 1: "calm", 2: "horror", 3: "funny"}


def _csv_path(subject: int, game: int) -> Path:
    """
    Build the path to a single subject/game CSV.

    The dataset nests files under a hidden-looking '.csv format' subdirectory.
    Centralising path construction here means we only fix it in one place if
    the folder structure ever changes.
    """
    subj_tag = f"S{subject:02d}"
    subj_dir = GAMEEMO_ROOT / f"({subj_tag})" / "Preprocessed EEG Data" / ".csv format"
    return subj_dir / f"{subj_tag}G{game}AllChannels.csv"


def load_recording(subject: int, game: int) -> np.ndarray:
    """
    Load one subject/game recording and return a clean (n_samples, 14) array.

    We drop 'Unnamed: 14' because it is an artefact of how the dataset was
    exported (trailing comma in every CSV row) — it contains no signal.
    Returning raw µV values without further scaling so callers can choose
    their own normalisation strategy.
    """
    path = _csv_path(subject, game)
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return df.values.astype(np.float32)   # (n_samples, 14)


def epoch_signal(signal: np.ndarray, label: int, subject: int, game: int):
    """
    Slice a continuous (n_samples, 14) signal into fixed-length epochs.

    50 % overlap doubles the number of training examples and reduces the
    chance that a meaningful event falls entirely at the boundary of a window.
    Any trailing samples that don't fill a complete window are discarded —
    keeping all epochs the same length avoids the need for padding.

    Returns
    -------
    epochs      : (n_epochs, 14, 256)  — channels-first for Conv1D/Conv2D
    labels      : (n_epochs,)
    subjects    : (n_epochs,)
    games       : (n_epochs,)
    """
    n_samples = signal.shape[0]
    starts = range(0, n_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES)

    epochs, labels, subjects, games = [], [], [], []
    for start in starts:
        window = signal[start : start + WINDOW_SAMPLES]   # (256, 14)
        epochs.append(window.T)                            # → (14, 256)
        labels.append(label)
        subjects.append(subject)
        games.append(game)

    return (
        np.array(epochs, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(subjects, dtype=np.int64),
        np.array(games, dtype=np.int64),
    )


def load_all(subjects=range(1, 29), games=range(1, 5), verbose=True) -> dict:
    """
    Load the full GAMEEMO dataset and return epoched arrays ready for ML.

    Iterating subject-first (then game) keeps subject_ids contiguous, which
    makes it easy to do subject-independent cross-validation later without
    shuffling the arrays.

    Parameters
    ----------
    subjects : iterable of int   — 1-indexed subject numbers (default: all 28)
    games    : iterable of int   — 1-indexed game numbers   (default: all 4)
    verbose  : bool              — print progress

    Returns
    -------
    dict with keys
        X           : (n_epochs, 14, 256)  float32
        y           : (n_epochs,)          int64   class labels
        subject_ids : (n_epochs,)          int64   1-indexed subject
        game_ids    : (n_epochs,)          int64   1-indexed game
    """
    np.random.seed(CONFIG["RANDOM_SEED"])  # seed here so any future shuffle is reproducible

    all_X, all_y, all_subj, all_game = [], [], [], []

    for subj in subjects:
        for game in games:
            label = LABEL_MAP[game]
            try:
                signal = load_recording(subj, game)
                X_ep, y_ep, s_ep, g_ep = epoch_signal(signal, label, subj, game)
                all_X.append(X_ep)
                all_y.append(y_ep)
                all_subj.append(s_ep)
                all_game.append(g_ep)
                if verbose:
                    print(f"  S{subj:02d} G{game} ({LABEL_NAMES[label]:>7s}): "
                          f"{signal.shape[0]:>6d} samples → {len(X_ep):>4d} epochs")
            except FileNotFoundError as e:
                print(f"  [WARN] missing: {e}")

    return {
        "X":           np.concatenate(all_X,    axis=0),
        "y":           np.concatenate(all_y,    axis=0),
        "subject_ids": np.concatenate(all_subj, axis=0),
        "game_ids":    np.concatenate(all_game, axis=0),
    }


# ---------------------------------------------------------------------------
# Summary when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading GAMEEMO dataset …\n")
    data = load_all(verbose=True)

    X, y = data["X"], data["y"]
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"X shape       : {X.shape}  (epochs × channels × samples)")
    print(f"y shape       : {y.shape}")
    print(f"dtype X / y   : {X.dtype} / {y.dtype}")
    print(f"Total epochs  : {len(y):,}")
    print()
    print("Epochs per class:")
    for cls, name in LABEL_NAMES.items():
        n = (y == cls).sum()
        print(f"  {cls} {name:>7s}: {n:>6,}  ({100*n/len(y):.1f}%)")
    print()
    print(f"Subjects      : {np.unique(data['subject_ids']).tolist()}")
    print(f"Window        : {WINDOW_SAMPLES} samples ({CONFIG['WINDOW_SEC']} s)")
    print(f"Step          : {STEP_SAMPLES} samples ({CONFIG['OVERLAP']*100:.0f}% overlap)")
    print("=" * 50)
