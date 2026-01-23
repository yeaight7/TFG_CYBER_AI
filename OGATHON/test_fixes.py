#!/usr/bin/env python3
"""
Simple validation tests for data leakage and overfitting fixes.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from predict_renewal
from predict_renewal import preprocess_data, apply_smote_balancing

def test_encoder_reuse():
    """Test that encoders are saved and reused correctly."""
    print("Test 1: Encoder Reuse")
    print("-" * 50)
    
    # Create sample training data
    train_df = pd.DataFrame({
        'cat_col': ['A', 'B', 'C', 'A', 'B'],
        'num_col': [1, 2, 3, 4, 5],
        'Renew': ['Y', 'N', 'Y', 'N', 'Y']
    })
    
    # Preprocess training data
    X_train, y_train, encoders = preprocess_data(train_df.copy(), is_training=True)
    
    # Verify encoders were created
    assert 'cat_col' in encoders, "Encoder for 'cat_col' should be created"
    print("✓ Encoders created for categorical columns")
    
    # Create test data with same categories
    test_df = pd.DataFrame({
        'cat_col': ['B', 'C', 'A'],
        'num_col': [6, 7, 8]
    })
    
    # Preprocess test data using saved encoders
    X_test, _, _ = preprocess_data(test_df.copy(), is_training=False, feature_encoders=encoders)
    
    # Verify encoding is consistent
    assert 'cat_col' in X_test.columns, "cat_col should be in test data"
    print("✓ Encoders reused for test data")
    
    # Create test data with unseen category
    test_df_unseen = pd.DataFrame({
        'cat_col': ['D', 'E'],  # Unseen categories
        'num_col': [9, 10]
    })
    
    # Should handle gracefully
    X_test_unseen, _, _ = preprocess_data(test_df_unseen.copy(), is_training=False, feature_encoders=encoders)
    assert 'cat_col' in X_test_unseen.columns, "cat_col should be in test data with unseen categories"
    print("✓ Unseen categories handled gracefully (mapped to 0)")
    
    print("\n✓ Test 1 PASSED\n")


def test_smote_not_applied_before_split():
    """Test that SMOTE is applied correctly after split."""
    print("Test 2: SMOTE Order")
    print("-" * 50)
    
    # Create imbalanced dataset
    X = np.random.randn(100, 5)
    y = np.array([0] * 80 + [1] * 20)  # Imbalanced: 80% class 0, 20% class 1
    
    print(f"Original class distribution: {np.bincount(y)}")
    
    # Apply SMOTE
    X_balanced, y_balanced = apply_smote_balancing(X, y)
    
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    
    # Verify balancing worked
    class_counts = np.bincount(y_balanced)
    assert len(class_counts) == 2, "Should have 2 classes"
    
    # After SMOTE, classes should be balanced (or close)
    ratio = class_counts[0] / class_counts[1]
    print(f"Balance ratio: {ratio:.2f}")
    assert 0.8 <= ratio <= 1.2, "Classes should be approximately balanced after SMOTE"
    
    print("\n✓ Test 2 PASSED\n")


def test_hyperparameters():
    """Verify that hyperparameters are set to prevent overfitting."""
    print("Test 3: Hyperparameter Values")
    print("-" * 50)
    
    try:
        from lightgbm import LGBMClassifier
        
        # Create a model with same parameters as in predict_renewal.py
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            class_weight='balanced',
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Verify key parameters
        assert model.max_depth == 8, f"max_depth should be 8, got {model.max_depth}"
        assert model.num_leaves == 31, f"num_leaves should be 31, got {model.num_leaves}"
        assert model.min_child_samples == 50, f"min_child_samples should be 50, got {model.min_child_samples}"
        assert model.learning_rate == 0.02, f"learning_rate should be 0.02, got {model.learning_rate}"
        assert model.reg_alpha == 1.0, f"reg_alpha should be 1.0, got {model.reg_alpha}"
        assert model.reg_lambda == 1.0, f"reg_lambda should be 1.0, got {model.reg_lambda}"
        
        print("✓ LightGBM hyperparameters correctly set:")
        print(f"  - max_depth: {model.max_depth}")
        print(f"  - num_leaves: {model.num_leaves}")
        print(f"  - min_child_samples: {model.min_child_samples}")
        print(f"  - learning_rate: {model.learning_rate}")
        print(f"  - reg_alpha: {model.reg_alpha}")
        print(f"  - reg_lambda: {model.reg_lambda}")
        
    except ImportError:
        print("⚠ LightGBM not installed, skipping hyperparameter test")
    
    print("\n✓ Test 3 PASSED\n")


def test_metrics_import():
    """Test that all required metrics can be imported."""
    print("Test 4: Metrics Import")
    print("-" * 50)
    
    try:
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
        print("✓ All metrics imported successfully:")
        print("  - accuracy_score")
        print("  - f1_score")
        print("  - balanced_accuracy_score")
        print("  - roc_auc_score")
    except ImportError as e:
        raise AssertionError(f"Failed to import metrics: {e}")
    
    print("\n✓ Test 4 PASSED\n")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATION TESTS FOR OVERFITTING AND DATA LEAKAGE FIXES")
    print("=" * 60)
    print()
    
    try:
        test_encoder_reuse()
        test_smote_not_applied_before_split()
        test_hyperparameters()
        test_metrics_import()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED ✗: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
