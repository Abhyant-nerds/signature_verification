import numpy as np

from signature_verification.evaluation import calibrate_threshold, evaluate_scores


def test_calibration_and_metrics_on_separable_scores():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    calibration = calibrate_threshold(labels, scores)
    metrics = evaluate_scores(labels, scores, calibration["threshold"])
    assert calibration["eer"] == 0
    assert metrics["far"] == 0
    assert metrics["frr"] == 0
