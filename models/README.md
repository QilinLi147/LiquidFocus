# Model files

The complete model architecture is stored as source code in
`liquidfocus/`. The training command writes one `model.pt` file for each
participant and fold under the requested output directory.

Pretrained weights are not included. Train with the supplied scripts to
produce checkpoints containing the model state, configuration and fitted
normalisation. Keep each checkpoint with its participant/fold metrics.

Use `scripts/evaluate.py` to evaluate a checkpoint on its recorded test split.
Only load checkpoints from a trusted source.
