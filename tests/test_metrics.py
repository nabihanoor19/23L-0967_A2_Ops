import numpy as np

from src.metrics import best_threshold, metrics_at


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics_at(y, p, 0.5)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (2, 0, 0, 2)


def test_best_threshold_lies_between_the_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < best_threshold(y, p) <= 0.8