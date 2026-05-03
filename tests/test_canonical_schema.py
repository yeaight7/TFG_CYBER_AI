import pytest
import numpy as np
import pandas as pd
from src.canonical_schema import (
    FEATURES_CANON,
    NUM_CANONICAL_FEATURES,
    NUM_OBSERVATION_FEATURES,
    CICIDS2017_TO_CANON,
    map_to_canonical
)

def test_canonical_features_length():
    assert len(FEATURES_CANON) == 76
    assert NUM_CANONICAL_FEATURES == 76
    assert NUM_OBSERVATION_FEATURES == 152
    assert len(set(FEATURES_CANON)) == 76

def test_map_to_canonical_mask_logic():
    df = pd.DataFrame({
        "Flow Duration": [100, np.nan, 300],
        "Unknown Col": [1, 2, 3]
    })
    
    res = map_to_canonical(df, CICIDS2017_TO_CANON, imputation_value=-1.0)
    
    assert res.X.shape == (3, 76)
    assert res.mask.shape == (3, 76)
    assert res.combined.shape == (3, 152)
    
    np.testing.assert_array_equal(res.X[:, 0], [100, -1, 300])
    np.testing.assert_array_equal(res.mask[:, 0], [1, 0, 1])
    
    assert np.all(res.X[:, 1:] == -1.0)
    assert np.all(res.mask[:, 1:] == 0.0)
    
    assert res.n_present == 1
    assert res.n_missing == 75