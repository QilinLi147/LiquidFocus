from __future__ import annotations

import numpy as np

from liquidfocus.data import FeatureArchive, protocol_folds
from liquidfocus.v163_stable_evidence import exact_stable_evidence


def test_seed_partitions_are_disjoint() -> None:
    session = np.repeat(np.arange(3), 15)
    trial = np.tile(np.arange(15), 3)
    archive = FeatureArchive(
        x=np.zeros((45, 62, 10, 5), dtype=np.float32),
        y=(trial % 3).astype(np.int64),
        subject=np.zeros(45, dtype=np.int64),
        session=session.astype(np.int64),
        trial=trial.astype(np.int64),
        emotion=None,
        metadata={"dataset": "seed"},
    )
    fold = protocol_folds(archive, "seed", [0])[0]
    assert len(fold.train) == 30
    assert len(fold.validation) == 3
    assert len(fold.test) == 12
    assert not set(fold.train) & set(fold.validation)
    assert not set(fold.train) & set(fold.test)
    assert not set(fold.validation) & set(fold.test)


def test_stable_feature_dimensions() -> None:
    evidence = exact_stable_evidence(np.zeros((2, 62, 10, 5), dtype=np.float32))
    assert evidence.channel.shape == (2, 620)
    assert evidence.relation.shape == (2, 315)
    assert evidence.combined.shape == (2, 935)
