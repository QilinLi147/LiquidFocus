# Data input contract

Raw EEG recordings are not redistributed in this package. Obtain SEED, SEED-IV,
SEED-V, MPED, or FACED from their official providers and comply with the
provider's access and licence terms.

- [SEED, SEED-IV and SEED-V](https://bcmi.sjtu.edu.cn/home/seed/)
- [MPED](https://github.com/Tengfei000/MPED)
- [FACED on NEMAR](https://nemar.org/dataset/nm000112)

The common four-dataset training entry accepts one compressed NumPy archive
with these arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `x` | `[N,62,10,5]` or `[N,62,50]` | differential-entropy features |
| `y` | `[N]` | zero-based target class |
| `subject` | `[N]` | zero-based participant index |
| `session` | `[N]` | zero-based session index |
| `trial` | `[N]` | zero-based trial index within the dataset protocol |
| `emotion` | `[N]`, MPED only | original seven-emotion trial label |

Band order is delta, theta, alpha, beta, gamma. Channel order is defined in
`liquidfocus/geometry.py`. `scripts/prepare_dataset.py` validates and
converts a provider-derived archive to this contract.

Keep participant, session and trial identifiers from the dataset metadata;
do not infer them from the order of files. A source feature archive must use
the stated channel order and contain all examples needed for the protocol.
The converter does not extract features from raw SEED-family or MPED EEG.
It performs array validation and reshaping, while the training entry applies
the participant-specific training/development/test partitions implemented in
`liquidfocus/data.py`.

FACED has a different montage. `scripts/preprocess_faced.py` writes
participant files shaped `[N,30,10,5]`; the schema adapter in
`liquidfocus/analysis/schema.py` supplies the corresponding model geometry.
The helper uses the provider's three valence categories (`negative`,
`neutral`, `positive`) and preserves the video index and sequence offset.
Its output is not the 62-channel archive consumed by `scripts/train.py`,
and the helper does not implement the nine-emotion FACED task.
