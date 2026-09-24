# LiquidFocus

This package contains the paper-facing LiquidFocus model, the four-dataset
participant protocol, train/development selection followed by fresh union
refitting, one-shot test evaluation, and the analysis operators used for state
transitions, policy quality, localisation, robustness, and latency profiling.

LiquidFocus separates a stable regional reference from a bounded, adaptive
liquid state. A prediction-decoupled policy then selects a region and, if
useful, one observed child channel as localised evidence.

This implementation accompanies *Dual-Timescale Fusion in Liquid Neural
Networks with Prediction-Decoupled Hierarchical Evidence Localisation for EEG
Emotion Recognition*. The repository contains model source, training and
evaluation entry points, data-preparation utilities, configurations and tests.

## Installation

Python 3.9 or newer is required. Create an isolated environment and install:

```bash
git clone https://github.com/QilinLi147/LiquidFocus.git
cd LiquidFocus
python -m pip install -e .
```

For FACED BDF preprocessing, install the optional dependency:

```bash
python -m pip install -e '.[faced]'
```

## Data preparation

Raw recordings are not included. First compute differential-entropy features
from data obtained through the official dataset provider. Supply a compressed
NumPy file with `x`, `y`, `subject`, `session`, and `trial`; MPED additionally
requires `emotion`, the original seven-emotion trial label.

Convert and validate the common archive:

```bash
python scripts/prepare_dataset.py \
  --dataset seed \
  --input /path/to/provider_features.npz \
  --output data/seed_features.npz
```

Use `--subject-offset`, `--session-offset`, or `--trial-offset` when the source
uses one-based indices. The complete array contract and channel ordering are
documented in `data/README.md`.

The optional FACED preprocessing entry operates directly on its BIDS BDF
layout and extracts 30-channel, three-valence-category sequences:

```bash
python scripts/preprocess_faced.py \
  --bids-root /path/to/faced/raw_bids \
  --participant sub-001 \
  --output data/faced/sub-001.npz
```

FACED uses a different montage. `liquidfocus.analysis.schema` provides the
corresponding geometry adapter. This helper is separate from the four-dataset
benchmark command and does not implement a nine-emotion FACED evaluation.

## Training

Train one participant and fold:

```bash
python scripts/train.py \
  --dataset seed \
  --archive data/seed_features.npz \
  --subject 0 \
  --fold 0 \
  --output runs/seed
```

Omit `--subject` and `--fold` to run every participant/fold in the archive.
The default process uses the development partition to choose the epoch,
residual scale, and STOP threshold. It then reinitialises the model, fits all
data-dependent transforms on the training-plus-development union, trains for
the selected epoch count, and evaluates the test partition once.

Each run writes:

- `model.pt`: model parameters, configuration, normalisation, and split
  identity;
- `metrics.json`: accuracy, balanced accuracy, macro-F1, test confusion
  counts, selection history and refit history;
- `summary.json` in the requested output directory: participant-level
  metrics, their mean and their sample standard deviation. For MPED, test
  confusion counts are pooled across the four folds of each participant
  before calculating these metrics. Participants receive equal weight.

Scores are stored on a 0–1 scale; multiply by 100 to report percentages.
Standard deviation is calculated across participants, not across folds or
test windows. A single-participant run records a standard deviation of zero.

Re-evaluate a saved model on the same test split:

```bash
python scripts/evaluate.py \
  --archive data/seed_features.npz \
  --checkpoint runs/seed/subject_00/fold_00/model.pt
```

Load only checkpoints from a trusted source.

## Model implementation

The public construction entry is:

```python
from liquidfocus import build_model
from liquidfocus.training import model_config

model = build_model(classes=3, config=model_config())
```

The installable distribution and canonical Python package are both named
`liquidfocus`. The released model class is `EvidenceDecoupledLiquidFocus`.
An isolated `liquidfocus_eeg` compatibility entry keeps earlier import paths
and serialised class references working. It points to the same implementation;
model parameter names, tensor values and checkpoint dictionaries are unchanged.
New scripts and examples use only the canonical package.

The source files retain the research-version filenames needed to load the
current model state without changing parameter names. Their responsibilities
are:

- `compact.py`: encoder/router/refiner hierarchy and joint objective;
- `v163_stable_evidence.py`: causal node and lagged region-relation features;
- `v164_stable_liquid.py`: stable-evidence observation;
- `v182_dual_timescale.py`: slow/fast regional liquid states;
- `v183_anchored_liquid.py`, `v185_exact_anchor_liquid.py`, and
  `v185_anchor_fit.py`: train-fitted regional anchor and bounded residual;
- `v186_evidence_decoupled.py`: fixed primary prediction and evidence-only
  region-to-channel route;
- `analysis/state_gates.py`: node, relation, slow-state, and fast-state traces;
- `analysis/sparse_inference.py`: literal sparse STOP/REGION/CHANNEL execution.

## Analysis utilities

`liquidfocus/analysis/` also includes exact three-component Shapley and
pair-interaction calculations, constrained policy return and regret metrics,
matched localisation interventions, missing-channel/time/band perturbations,
participant-level statistics, and batch-one latency profiling. These modules
operate on model outputs and participant-level arrays; they do not contain
paper results.

## Tests

Install the test extra and run:

```bash
python -m pip install -e '.[test]'
pytest
```

The included tests verify the 620-node/315-relation feature dimensions,
participant split disjointness, state-trace dimensions, and the invariant that
the evidence path cannot change the reported emotion logits. Reporting tests
check pooled-fold participant metrics against scikit-learn and reject
duplicate folds or incomplete multi-fold records.

## Data and trained parameters

- No raw third-party EEG data are redistributed.
- The four-dataset converter starts from provider-derived differential-entropy
  features, not from raw recordings. Dataset access links and input details
  are in [data/README.md](data/README.md).
- Pretrained weights and participant-level study results are not included.
  The training command creates checkpoints for the requested participants.
- Analysis modules provide reusable operators; they are not a stored copy of
  the manuscript's result tables or a one-command rerun of every experiment.

## Licence

The code is released under the [MIT licence](LICENSE). Third-party datasets
remain subject to their providers' access and licence conditions.
