import pandas as pd
import numpy as np
import pytest
from scripts.predict_real_traffic_v2 import (
    maybe_convert_time_units,
    compute_diagnostics,
    compute_truth_metrics,
)

def test_maybe_convert_time_units():
    df_sec = pd.DataFrame({"flow_duration": [0.5, 0.2], "flow_iat_mean": [0.1, 0.1]})
    df_usec = pd.DataFrame({"flow_duration": [500000, 200000], "flow_iat_mean": [100000, 100000]})
    
    out_sec = maybe_convert_time_units(df_sec)
    np.testing.assert_allclose(out_sec["flow_duration"].values, [500000., 200000.])
    
    out_usec = maybe_convert_time_units(df_usec)
    np.testing.assert_allclose(out_usec["flow_duration"].values, [500000, 200000])

def test_compute_diagnostics():
    X = np.zeros((10, 152))
    X[0, 0] = 11.0 
    
    names = [f"f{i}" for i in range(152)]
    diag = compute_diagnostics(X, names)
    
    assert diag["z_abs_max"] == 11.0
    assert diag["z_gt10_count"] == 1
    assert diag["top_features"][0]["feature"] == "f0"

def test_compute_truth_metrics_row_mismatch_raises():
    # C5: predictions must align 1:1 with the flows DataFrame.
    df = pd.DataFrame({"truth_y": [1, 0, 1]})
    y_pred = np.array([1, 0])  # wrong length
    with pytest.raises(ValueError):
        compute_truth_metrics(df, y_pred)

def test_compute_truth_metrics_aligned_ok():
    # Sanity: matching lengths still compute metrics.
    df = pd.DataFrame({"truth_y": [1, 0, 1, 0]})
    y_pred = np.array([1, 0, 0, 0])
    out = compute_truth_metrics(df, y_pred)
    assert out is not None
    assert out["n_labeled_flows"] == 4
