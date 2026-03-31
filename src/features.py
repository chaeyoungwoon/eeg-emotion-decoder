"""
Feature extraction for EEG emotion decoding.

Two complementary representations are implemented:

1. Band Power (PSD-based): captures how much energy lives in each frequency
   band using Welch's method. Robust to noise because it averages over many
   periodogram segments. Good baseline and widely used in BCI literature.

2. Differential Entropy (DE): a theoretically grounded alternative to band
   power. DE of a Gaussian signal = 0.5*log(2πe*σ²), but we bandpass-filter
   first so the signal is approximately band-limited before computing variance.
   DE has been shown to outperform raw band power on SEED/DEAP-style datasets
   because it is more sensitive to changes in the distribution shape, not just
   amplitude.
"""

import time
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch, butter, sosfilt

# ---------------------------------------------------------------------------
# Config — single source of truth for frequency band definitions
# ---------------------------------------------------------------------------
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
N_BANDS = len(BANDS)   # 5


# ---------------------------------------------------------------------------
# Band Power via Welch PSD
# ---------------------------------------------------------------------------

def _band_power_epoch(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute mean PSD in each frequency band for one epoch.

    Welch's method is preferred over a plain FFT because it reduces variance
    by averaging over overlapping segments (nperseg=256 uses the full window,
    so here it equals a single periodogram — but the API is consistent with
    longer epochs if we ever increase window size).

    Parameters
    ----------
    epoch  : (n_channels, n_samples)
    sfreq  : sampling frequency in Hz

    Returns
    -------
    features : (n_channels * n_bands,)  — channels vary slowest (C×B layout)
    """
    n_channels = epoch.shape[0]
    freqs, psd = welch(epoch, fs=sfreq, nperseg=epoch.shape[1], axis=-1)
    # psd shape: (n_channels, n_freqs)

    features = np.empty(n_channels * N_BANDS, dtype=np.float32)
    for b_idx, (lo, hi) in enumerate(BANDS.values()):
        mask = (freqs >= lo) & (freqs < hi)
        # Mean PSD in band — more stable than sum because band widths differ
        band_power = psd[:, mask].mean(axis=-1)  # (n_channels,)
        features[b_idx * n_channels : (b_idx + 1) * n_channels] = band_power

    return features


def extract_bandpower(X: np.ndarray, sfreq: float = 128) -> np.ndarray:
    """
    Extract Welch band-power features for all epochs.

    Output layout: [delta_ch0..delta_ch13, theta_ch0..theta_ch13, ...]
    This groups by band (not channel) so that band-wise slicing is O(1)
    without reshaping — convenient for ablation studies.

    Parameters
    ----------
    X     : (n_epochs, n_channels, n_samples)
    sfreq : sampling frequency in Hz

    Returns
    -------
    features : (n_epochs, n_channels * n_bands)  e.g. (33264, 70)
    """
    n_epochs, n_channels, _ = X.shape
    out = np.empty((n_epochs, n_channels * N_BANDS), dtype=np.float32)
    for i in range(n_epochs):
        out[i] = _band_power_epoch(X[i], sfreq)
    return out


# ---------------------------------------------------------------------------
# Differential Entropy
# ---------------------------------------------------------------------------

def _make_bandpass(lo: float, hi: float, sfreq: float, order: int = 4):
    """
    Design a Butterworth bandpass filter and return second-order sections.

    Using SOS (second-order sections) instead of BA coefficients avoids
    numerical instability for higher-order filters — especially important for
    narrow bands like delta (1-4 Hz) at 128 Hz where poles can be very
    close to the unit circle.
    """
    nyq = sfreq / 2.0
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sos


# Pre-compute filters once — building them inside the per-epoch loop would
# waste ~90% of feature extraction time.
_FILTERS = {
    name: _make_bandpass(lo, hi, sfreq=128)
    for name, (lo, hi) in BANDS.items()
}


def _de_epoch(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute differential entropy per band per channel for one epoch.

    DE for a Gaussian: H = 0.5 * log(2πe * σ²)
    We bandpass-filter the epoch into each band first so that σ² captures
    only within-band variance. Without filtering, broadband variance would
    dominate and the band-specific signal would be lost.

    Parameters
    ----------
    epoch  : (n_channels, n_samples)
    sfreq  : sampling frequency in Hz (used only for filter rebuild if needed)

    Returns
    -------
    features : (n_channels * n_bands,)
    """
    n_channels = epoch.shape[0]
    features = np.empty(n_channels * N_BANDS, dtype=np.float32)

    for b_idx, (name, _) in enumerate(BANDS.items()):
        sos = _FILTERS[name]
        filtered = sosfilt(sos, epoch, axis=-1)      # (n_channels, n_samples)
        var = filtered.var(axis=-1) + 1e-10          # avoid log(0)
        de = 0.5 * np.log(2 * np.pi * np.e * var)   # (n_channels,)
        features[b_idx * n_channels : (b_idx + 1) * n_channels] = de

    return features


def extract_de(X: np.ndarray, sfreq: float = 128) -> np.ndarray:
    """
    Extract differential entropy features for all epochs.

    Output layout mirrors extract_bandpower for easy comparison:
    [delta_ch0..delta_ch13, theta_ch0..theta_ch13, ...]

    Parameters
    ----------
    X     : (n_epochs, n_channels, n_samples)
    sfreq : sampling frequency in Hz

    Returns
    -------
    features : (n_epochs, n_channels * n_bands)  e.g. (33264, 70)
    """
    n_epochs = X.shape[0]
    out = np.empty((n_epochs, X.shape[1] * N_BANDS), dtype=np.float32)
    for i in range(n_epochs):
        out[i] = _de_epoch(X[i], sfreq)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Add repo root to path so we can import loader from any working directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.loader import load_all, CONFIG

    RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    t0 = time.time()
    data = load_all(verbose=False)
    X, y, subject_ids = data["X"], data["y"], data["subject_ids"]
    print(f"  Loaded in {time.time()-t0:.1f}s  —  X: {X.shape}, y: {y.shape}\n")

    sfreq = CONFIG["SFREQ"]

    # Band power
    print("Extracting band-power features …")
    t0 = time.time()
    feat_bp = extract_bandpower(X, sfreq=sfreq)
    t_bp = time.time() - t0
    print(f"  Shape : {feat_bp.shape}")
    print(f"  Time  : {t_bp:.1f}s  ({t_bp/len(X)*1000:.2f} ms/epoch)\n")

    # Differential entropy
    print("Extracting differential entropy features …")
    t0 = time.time()
    feat_de = extract_de(X, sfreq=sfreq)
    t_de = time.time() - t0
    print(f"  Shape : {feat_de.shape}")
    print(f"  Time  : {t_de:.1f}s  ({t_de/len(X)*1000:.2f} ms/epoch)\n")

    # Sanity checks — NaN/Inf would silently break downstream classifiers
    for name, arr in [("band_power", feat_bp), ("de", feat_de)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        print(f"  {name:>12s}  NaN={n_nan}  Inf={n_inf}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

    # Save
    print("\nSaving …")
    np.save(RESULTS_DIR / "features_bandpower.npy", feat_bp)
    np.save(RESULTS_DIR / "features_de.npy",        feat_de)
    np.save(RESULTS_DIR / "labels.npy",             y)
    np.save(RESULTS_DIR / "subject_ids.npy",        subject_ids)
    print(f"  Saved to {RESULTS_DIR}/")
    print("  features_bandpower.npy  features_de.npy  labels.npy  subject_ids.npy")
