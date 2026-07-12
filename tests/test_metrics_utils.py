import pytest

from src.metrics_utils import confusion_to_metrics


def test_basic_counts_and_rates():
    # tn=80, fp=5, fn=10, tp=5  -> total 100
    m = confusion_to_metrics(80, 5, 10, 5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (80, 5, 10, 5)
    assert abs(m["accuracy"] - 0.85) < 1e-9
    assert abs(m["recall_attack"] - (5 / 15)) < 1e-9
    assert abs(m["precision_attack"] - (5 / 10)) < 1e-9
    assert abs(m["fpr"] - (5 / 85)) < 1e-9
    assert abs(m["fnr"] - (10 / 15)) < 1e-9
    assert abs(m["balanced_accuracy"] - ((5 / 15 + 80 / 85) / 2)) < 1e-9
    assert abs(m["block_rate"] - (10 / 100)) < 1e-9


def test_perfect_classifier():
    m = confusion_to_metrics(50, 0, 0, 50)
    assert m["accuracy"] == 1.0
    assert m["mcc"] == 1.0
    assert m["fpr"] == 0.0 and m["fnr"] == 0.0


def test_zero_division_is_safe():
    m = confusion_to_metrics(0, 0, 0, 0)
    rate_keys = (
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "precision_benign",
        "recall_benign",
        "f1_benign",
        "specificity",
        "fpr",
        "fnr",
        "block_rate",
    )
    assert all(m[key] == 0.0 for key in rate_keys)


def test_nullable_policy_preserves_undefined_metrics_as_none():
    m = confusion_to_metrics(10, 0, 0, 0, undefined_metric_policy="null")

    assert m["accuracy"] == 1.0
    assert m["precision_attack"] is None
    assert m["recall_attack"] is None
    assert m["f1_attack"] is None
    assert m["balanced_accuracy"] is None
    assert m["mcc"] is None
    assert m["fnr"] is None
    assert m["precision_benign"] == 1.0
    assert m["recall_benign"] == 1.0
    assert m["fpr"] == 0.0

    empty = confusion_to_metrics(0, 0, 0, 0, undefined_metric_policy="null")
    assert all(empty[key] is None for key in empty if key not in {"tp", "tn", "fp", "fn"})


def test_nullable_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="undefined_metric_policy"):
        confusion_to_metrics(1, 0, 0, 1, undefined_metric_policy="missing")


def test_reward_total_matches_cost_matrix():
    rc = {"tp": 1.5, "fp": -2.0, "fn": -5.0, "omission": 0.0}
    m = confusion_to_metrics(80, 5, 10, 5, reward_config=rc)
    expected = 5 * 1.5 + 5 * (-2.0) + 10 * (-5.0) + 80 * 0.0
    assert abs(m["reward_total"] - expected) < 1e-9
    assert abs(m["reward_per_sample"] - expected / 100) < 1e-9
