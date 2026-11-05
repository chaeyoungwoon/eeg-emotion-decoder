# EEG Emotion Decoder — Project State

## What this project is
Phase 1 of a 2-phase BCI project. Phase 2 pipes decoded emotion into a real-time generative narrative engine via Claude API.

## Dataset
GAMEEMO — 28 subjects, 4 classes (boring/calm/horror/funny), 14-channel EEG at 128Hz, preprocessed CSVs in data/GAMEEMO/

## Completed
- src/loader.py — loads all subjects, epochs into (33264, 14, 256), perfectly balanced
- src/features.py — band power and DE features saved to results/metrics/
- src/models/baselines.py — XGBoost LOSO done
  - XGBoost + bandpower: acc=0.403, f1=0.377, auc=0.666
  - XGBoost + DE: acc=0.396, f1=0.365, auc=0.666
- src/models/eegnet.py — training now, 28 LOSO folds, MPS accelerated

## Next steps
1. Get EEGNet results
2. src/interpret.py — SHAP + EEGNet spatial filter topomap
3. app/streamlit_demo.py
4. README.md

## Environment
conda env: bci | Python 3.10 | Mac Apple Silicon
Run with: conda run -n bci python src/...
AlwAlwAlwAlwAlP_DUPLICATE_LIB_OK=TRUE
