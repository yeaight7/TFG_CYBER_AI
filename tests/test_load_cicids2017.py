from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import src.load_cicids2017 as lc
from src.load_cicids2017 import (
    CICIDSLoadConfig,
    SUBSAMPLE_METHOD_STRATIFIED_NESTED_PREFIX,
    _prepare_cicids_features,
    _sha256_of_array,
    _stratified_nested_prefix_indices,
    load_cicids2017_split,
)


def test_prepare_cicids_features_binary_labels():
    df = pd.DataFrame({
        "Flow Duration": [10, 20, 30],
        "Source IP": ["1.1.1.1", "2.2.2.2", "3.3.3.3"],
        "Label": ["BENIGN", "ATTACK", " BENIGN "]
    })

    cfg = CICIDSLoadConfig(label_col="Label", benign_value="BENIGN", drop_identifier_cols=True, use_canonical=True)
    X, y, feats = _prepare_cicids_features(df, cfg)

    np.testing.assert_array_equal(y, [0, 1, 0])
    assert "Source IP" not in feats
    assert len(feats) == 152
    assert X.shape == (3, 152)


# ──────────────────────────────────────────────────────────────
# train_max_rows: fixed test partition + train-only subsampling
# ──────────────────────────────────────────────────────────────

def _synthetic_df(n: int = 2000, n_attack: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = np.array(["BENIGN"] * (n - n_attack) + ["ATTACK"] * n_attack)
    labels = labels[rng.permutation(n)]
    return pd.DataFrame({
        "Flow Duration": rng.integers(1, 1000, n),
        "Total Fwd Packets": rng.integers(1, 100, n),
        "Label": labels,
    })


@pytest.fixture
def patched_loader(monkeypatch):
    df = _synthetic_df()
    monkeypatch.setattr(lc, "_list_csv_files", lambda root: [Path("synthetic.csv")])
    monkeypatch.setattr(lc, "_load_all_csvs", lambda csvs, cfg: df.copy())
    return df


def _split(**kwargs):
    defaults = dict(split_mode="random", preset="full", seed=42, scale=False, use_canonical=True)
    defaults.update(kwargs)
    return load_cicids2017_split(**defaults)


def test_nested_prefix_indices_nested_and_stratified():
    rng = np.random.default_rng(1)
    y = np.array([0] * 8000 + [1] * 2000, dtype=np.int64)[rng.permutation(10_000)]

    prev: set = set()
    for n in (1000, 2000, 5000):
        idx = _stratified_nested_prefix_indices(y, n, seed=42)
        assert len(idx) == n
        assert len(np.unique(idx)) == n
        assert np.all(np.diff(idx) > 0)  # sorted ascending
        n_attack = int(round(n * 0.2))
        assert int((y[idx] == 1).sum()) == n_attack
        assert int((y[idx] == 0).sum()) == n - n_attack
        assert prev.issubset(set(idx.tolist()))  # nesting
        prev = set(idx.tolist())


def test_nested_prefix_indices_deterministic():
    y = np.array([0] * 800 + [1] * 200, dtype=np.int64)
    a = _stratified_nested_prefix_indices(y, 100, seed=42)
    b = _stratified_nested_prefix_indices(y, 100, seed=42)
    c = _stratified_nested_prefix_indices(y, 100, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_train_max_rows_keeps_test_set_identical(patched_loader):
    X_tr_full, y_tr_full, X_te_full, y_te_full, _, _, meta_full = _split()
    X_tr_sub, y_tr_sub, X_te_sub, y_te_sub, _, _, meta_sub = _split(train_max_rows=800)

    np.testing.assert_array_equal(X_te_full, X_te_sub)
    np.testing.assert_array_equal(y_te_full, y_te_sub)
    assert meta_full["test_set_sha256"] == meta_sub["test_set_sha256"]
    assert meta_full["y_test_sha256"] == meta_sub["y_test_sha256"]
    assert len(y_tr_sub) == 800
    assert len(y_tr_full) == meta_sub["n_train_full"]


def test_train_max_rows_guards(patched_loader):
    with pytest.raises(ValueError, match="train_max_rows requires the full dataset"):
        _split(preset="fast", train_max_rows=800)
    with pytest.raises(ValueError, match="train_max_rows requires the full dataset"):
        _split(max_rows=1000, train_max_rows=800)
    with pytest.raises(ValueError, match="train_max_rows must be in"):
        _split(train_max_rows=0)
    with pytest.raises(ValueError, match="train_max_rows must be in"):
        _split(train_max_rows=1600)  # == n_train for the synthetic split


def test_metadata_new_keys(patched_loader):
    *_, meta = _split(train_max_rows=800)
    assert meta["train_max_rows"] == 800
    assert meta["n_train"] == 800
    assert meta["n_train_full"] == 1600
    assert meta["subsample_method"] == SUBSAMPLE_METHOD_STRATIFIED_NESTED_PREFIX
    assert meta["scale"] is False
    for key in ("test_set_sha256", "y_test_sha256", "train_set_sha256", "y_train_sha256"):
        assert isinstance(meta[key], str) and len(meta[key]) == 64

    *_, meta_full = _split()
    assert meta_full["train_max_rows"] is None
    assert meta_full["subsample_method"] is None
    assert meta_full["n_train_full"] == meta_full["n_train"] == 1600


def test_scale_true_refits_on_subsample(patched_loader):
    X_tr_raw, _, X_te_raw, _, _, _, _ = _split(train_max_rows=800)
    X_tr_sc, _, X_te_sc, _, scaler, _, meta = _split(train_max_rows=800, scale=True)

    assert meta["scale"] is True
    manual = StandardScaler().fit(X_tr_raw)
    np.testing.assert_allclose(scaler.mean_, manual.mean_)
    np.testing.assert_allclose(scaler.scale_, manual.scale_)
    np.testing.assert_allclose(X_te_sc, manual.transform(X_te_raw).astype(np.float32), rtol=1e-5)


def test_sha256_of_array_stable():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert _sha256_of_array(arr) == _sha256_of_array(arr.copy())
    assert _sha256_of_array(arr) != _sha256_of_array(arr.astype(np.float64))
    assert _sha256_of_array(arr) != _sha256_of_array(arr.reshape(4, 3))
    assert _sha256_of_array(arr) != _sha256_of_array(arr + 1)
