import pytest
import pandas as pd
import numpy as np
from src.load_cicids2017 import _prepare_cicids_features, CICIDSLoadConfig

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