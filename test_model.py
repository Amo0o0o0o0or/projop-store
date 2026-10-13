import numpy as np

from model import calculate_metrics, load_dataset, train_and_evaluate


def test_dataset_has_expected_shape_and_binary_target():
    features, target = load_dataset()

    assert features.shape == (569, 30)
    assert set(target.unique()) == {0, 1}
    assert target.sum() == 212


def test_metrics_treat_malignant_as_positive_class():
    metrics = calculate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.4, 0.9]))

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["recall"] == 0.5


def test_training_selects_and_evaluates_a_model():
    results = train_and_evaluate()

    assert results["model_name"] in {"Logistic regression", "Random forest"}
    assert len(results["y_test"]) > 0
    assert len(results["leaderboard"]) == 2
    assert np.isfinite(results["probabilities"]).all()