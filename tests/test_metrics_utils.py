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
    assert m["accuracy"] == 0.0
    assert m["mcc"] == 0.0
    assert m["precision_attack"] == 0.0


def test_reward_total_matches_cost_matrix():
    rc = {"tp": 1.5, "fp": -2.0, "fn": -5.0, "omission": 0.0}
    m = confusion_to_metrics(80, 5, 10, 5, reward_config=rc)
    expected = 5 * 1.5 + 5 * (-2.0) + 10 * (-5.0) + 80 * 0.0
    assert abs(m["reward_total"] - expected) < 1e-9
    assert abs(m["reward_per_sample"] - expected / 100) < 1e-9
