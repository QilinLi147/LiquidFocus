import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from liquidfocus.training import aggregate_reports, metrics_from_confusion


def report(subject, fold, truth, predicted):
    return {"dataset": "mped", "subject": subject, "fold": fold,
            "test_metrics": {"confusion_matrix": confusion_matrix(
                truth, predicted, labels=[0, 1, 2]).tolist()}}


def test_pooled_metrics_match_sklearn():
    truth = [0, 0, 1, 1, 2, 2, 2]
    predicted = [0, 1, 1, 2, 0, 2, 2]
    metrics = metrics_from_confusion(confusion_matrix(truth, predicted))
    assert metrics["accuracy"] == pytest.approx(accuracy_score(truth, predicted))
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, predicted))
    assert metrics["macro_f1"] == pytest.approx(f1_score(truth, predicted, average="macro"))


def test_folds_are_pooled_before_participant_mean_and_sample_sd():
    rows = [report(0, 0, [0, 1, 2], [0, 1, 2]),
            report(0, 1, [0, 1, 2] * 3, [1, 2, 0] * 3),
            report(1, 0, [0, 1, 2], [0, 1, 2])]
    summary = aggregate_reports(rows)
    assert summary["participants"] == 2
    assert summary["runs"] == 3
    assert summary["participant_metrics"][0]["accuracy"] == 0.25
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert summary[metric]["mean"] == pytest.approx(0.625)
        assert summary[metric]["std"] == pytest.approx(np.std([0.25, 1.0], ddof=1))


def test_duplicate_folds_are_rejected():
    row = report(0, 0, [0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_reports([row, row])


def test_missing_confusion_is_not_replaced_by_fold_average():
    rows = [{"dataset": "mped", "subject": 0, "fold": fold,
             "test_metrics": {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5}}
            for fold in range(2)]
    with pytest.raises(ValueError, match="confusion"):
        aggregate_reports(rows)
