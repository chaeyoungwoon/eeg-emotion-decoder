# EEG Emotion Decoder -- Phase 1: Neural Affect Classification for Real-Time Adaptive BCI Systems

A cross-subject EEG emotion classifier trained on the GAMEEMO dataset, built as the first stage of a closed-loop brain-computer interface for emotion-adaptive environments.

---

## Motivation

I am a data science student at UCSD and I want to work on BCI for fully immersive VR. The concept comes from full-dive VR -- the kind depicted in Sword Art Online -- where the environment is not a screen you look at but a world you inhabit. The core problem with building something like that is not graphics or haptics: it is that a truly immersive environment has to respond to your actual mental and emotional state, not just your button presses. This project attacks the decoding half of that problem -- how do you take raw brain signals and reliably classify what someone is feeling, across different people, in real time?

---

## The Closed-Loop Vision

The full system looks like this:

```
EEG Headset -> Emotion Classifier -> Generative Narrative Engine -> Adaptive Environment -> User -> EEG Headset -> ...
```

The user wears an EEG headset. The classifier reads brain signals and outputs an emotion state. That state feeds into a generative narrative engine, which produces adaptive content -- story beats, environmental changes, music -- conditioned on what the user is actually feeling. The environment shifts, the user's emotional state changes, the classifier picks that up, and the loop continues.

Phase 1 (this repository) is the classifier. Phase 2, currently in development, pipes the classifier output into a Claude API call that generates real-time adaptive story beats conditioned on the detected emotion state.

---

## Dataset: GAMEEMO

GAMEEMO is a publicly available EEG emotion dataset collected by Karakas (2020).

- 28 subjects wore a 14-channel Emotiv EPOC EEG headset
- Each subject played 4 video games, each designed to elicit a specific emotional response:
  - **G1: Train Sim World** -- boring/neutral
  - **G2: Unravel** -- calm/relaxed
  - **G3: Slender: The Arrival** -- horror/fear
  - **G4: Goat Simulator** -- funny/amusement
- After each game, subjects self-reported their emotional state to confirm the intended affect was induced
- EEG recorded at 128 Hz; recordings are approximately 5 minutes per game per subject
- Data was preprocessed by the dataset authors (artifact removal, bandpass filtering)

---

## Data Format

Each CSV contains a time series of EEG readings across 14 electrode channels, sampled at 128 Hz. One row equals 1/128th of a second.

**The 14 channels follow the 10-20 international electrode placement system:**

| Channel | Region | Notes |
|---------|---------|-------|
| AF3, AF4 | Prefrontal | Anterior frontal, overlies prefrontal cortex -- involved in executive function, emotional regulation |
| F3, F4 | Frontal | Overlies frontal lobe -- associated with motor planning and frontal alpha asymmetry, a marker of emotional valence |
| F7, F8 | Frontotemporal | Junction of frontal and temporal lobes -- linked to language and affective processing |
| FC5, FC6 | Frontocentral | Between frontal and central regions, overlies motor cortex and supplementary motor area |
| T7, T8 | Temporal | Over temporal lobes -- involved in auditory processing, memory, and emotional experience |
| P7, P8 | Parietal | Posterior parietal cortex -- spatial processing, attention, sensory integration |
| O1, O2 | Occipital | Primary visual cortex -- visual processing, alpha rhythms prominent here |

Each value is a floating-point microvolt reading, centered near zero after preprocessing. Typical EEG amplitude is on the order of 10--100 microvolts.

---

## Pipeline

```
Raw CSVs
   |
   v
Epoching (2s windows, 50% overlap)
   |
   v
Feature Extraction (band power or differential entropy per channel per band)
   |
   v
Model Training
   |
   v
LOSO Cross-Subject Evaluation
```

**Epoching:** Continuous EEG is segmented into 2-second windows with 50% overlap, producing 256-sample epochs per window across 14 channels. This yields 33,264 epochs total, balanced across four classes.

**Feature Extraction:** For each epoch, spectral features are computed in five frequency bands across all 14 channels. Two feature sets are evaluated: band power and differential entropy.

**Evaluation:** Leave-one-subject-out (LOSO) cross-validation. The model trains on 27 subjects and tests on the held-out subject, rotating through all 28 subjects. Reported metrics are averaged across all folds.

---

## Models and Results

| Model | Features | Accuracy | Macro F1 | AUC-ROC |
|-------|----------|----------|----------|---------|
| XGBoost | Band Power | 0.403 | 0.377 | 0.666 |
| XGBoost | Differential Entropy | 0.396 | 0.365 | 0.666 |
| EEGNet | Raw epochs | in progress | in progress | in progress |
| Chance baseline | -- | 0.250 | 0.250 | 0.500 |

Chance baseline is 25% (4 classes, perfectly balanced dataset). All evaluation is cross-subject LOSO -- not within-subject. This distinction matters and is discussed in the next section.

---

## Feature Extraction

**Band Power** is the average signal power within a specific frequency range, computed via Welch's method. It captures how active a particular brain rhythm is during an epoch.

**Differential Entropy (DE)** is the entropy of a continuous probability distribution. For a Gaussian-distributed band-filtered signal, DE simplifies to the log of the variance in that band. It is more discriminative than raw band power for emotion classification and is the feature used in several state-of-the-art EEG affective computing systems.

Both features are computed for each of the five canonical EEG frequency bands:

| Band | Hz Range | Cognitive associations |
|------|----------|------------------------|
| Delta | 1--4 Hz | Deep sleep, unconscious processing |
| Theta | 4--8 Hz | Drowsiness, light sleep, meditative states, working memory |
| Alpha | 8--13 Hz | Relaxed wakefulness, idle cortex; suppressed under stress and cognitive load |
| Beta | 13--30 Hz | Active thinking, focus, alertness, anxiety |
| Gamma | 30--50 Hz | High-level processing, binding of sensory features, active engagement |

Alpha suppression is a reliable marker of arousal and stress -- alpha power drops when the brain shifts from idle to engaged. Frontal alpha asymmetry (relative left vs. right frontal alpha) correlates with emotional valence. Gamma activity increases during active cognitive and emotional processing. These physiological relationships are why frequency-domain features generalize better than raw time-series for emotion classification.

---

## Cross-Subject Generalization

This is the hard problem in EEG-based BCI. EEG signals vary substantially between individuals -- the same mental state produces different spatial and spectral signatures depending on head geometry, electrode contact quality, and individual neural organization. A model trained and tested on the same person can exploit those individual quirks and report inflated accuracy. That accuracy is not meaningful for a real product, which must work on a user it has never seen before.

Most published EEG emotion recognition work reports within-subject accuracy. Numbers like 80--90% appear frequently in the literature and are mostly artifacts of this evaluation choice. Cross-subject LOSO is the meaningful benchmark: it tests whether the model has learned something about the relationship between brain states and emotions in general, not just for one person.

This project evaluates cross-subject only. The 40% accuracy from XGBoost baselines is lower than numbers often cited in papers, but it reflects a harder and more honest evaluation. Closing this gap -- through better features, domain adaptation, or subject-invariant representations -- is the central technical problem.

---

## Interpretability

SHAP channel importance analysis and EEGNet spatial filter visualizations are in development (see `src/interpret.py`). The goal is to identify which electrodes and which frequency bands are driving classification, and to check whether the model's learned representations align with known neuroscience (frontal asymmetry for valence, temporal activity for arousal, etc.). Visualizations will include topographic maps of electrode importance overlaid on standard scalp diagrams.

---

## Phase 2: Adaptive Narrative Engine (In Development)

The second phase of this project connects the classifier to a generative language model. The architecture:

```
EEG stream (live or replay)
   |
   v
Emotion Classifier (Phase 1)
   |
   v
Detected emotion state (one of: boring / calm / horror / funny)
   |
   v
Claude API call with emotion state as context
   |
   v
Generated story beat conditioned on emotional state
   |
   v
Display to user
   |
   v
(user's emotion shifts) -> back to EEG stream
```

The feedback loop is the interesting part. The generated narrative is designed to modulate the user's emotional state -- a calm scene when the classifier detects fear, an escalating scene when it detects boredom. The classifier then picks up the shift, which changes the next generation. This is the core architecture of an emotion-adaptive environment. The goal is not a system that responds to emotion after the fact, but one that participates in a continuous loop with the user's mental state.

---

## Setup and Reproduction

```bash
git clone <repo>
conda create -n bci python=3.10
conda activate bci
pip install -r requirements.txt

# Download GAMEEMO dataset -- see data/README.md for instructions

python src/loader.py              # verify data loads correctly
python src/features.py            # extract band power and DE features
python src/models/baselines.py    # run XGBoost LOSO baselines
python src/models/eegnet.py       # train EEGNet (MPS-accelerated on Apple Silicon)
```

Set `KMP_DUPLICATE_LIB_OK=TRUE` if you encounter library conflicts on macOS with MPS acceleration.

---

## References

- Karakas, M. A. (2020). *GAMEEMO EEG Dataset*. Mendeley Data. https://doi.org/10.17632/f4h3ddbdmf.1
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
- Duan, R. N., Zhu, J. Y., & Lu, B. L. (2013). Differential entropy feature for EEG-based emotion classification. *2013 6th International IEEE/EMBS Conference on Neural Engineering (NER)*, 81--84.
