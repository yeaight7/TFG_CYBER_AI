import pytest
import pandas as pd
import numpy as np
from scripts.predict_real_traffic_v2 import maybe_convert_time_units, compute_diagnostics

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