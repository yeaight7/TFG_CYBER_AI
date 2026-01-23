#!/usr/bin/env python3
"""
Predictive Model for Renewal Approval/Rejection
This script trains a classification model to predict if a renewal should be approved (Y) or rejected (N).
Optimized for accuracy with efficient resource usage and handling of imbalanced datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM (faster and more efficient)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Try to import SMOTE for oversampling minority class
try:
    from imblearn.over_sampling import SMOTE  # type: ignore[import-not-found]
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "ia_data", "training_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "ia_data", "test_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "predictions.txt")


def clean_numeric_column(series):
    """Clean numeric columns that may have unusual formatting."""
    if series.dtype == object:
        # Replace dots used as thousand separators and handle decimal commas
        cleaned = series.astype(str).str.replace(r'\.(?=\d{3})', '', regex=True)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    return series


def preprocess_data(df, is_training=True, feature_encoders=None):
    """Preprocess the dataframe for model training/prediction.
    
    Args:
        df: DataFrame to preprocess
        is_training: Whether this is training data (True) or test data (False)
        feature_encoders: Dictionary of LabelEncoders for categorical features.
                         If None and is_training=True, new encoders will be created.
                         If provided and is_training=False, existing encoders will be reused.
    
    Returns:
        Tuple of (preprocessed_df, target, feature_encoders)
    """
    
    # Drop non-predictive columns
    columns_to_drop = ['DateAlt', 'KeyMed', 'KeyEnf']  # Date and key columns
    
    if is_training:
        columns_to_drop.append('Renew')
        target = df['Renew'].copy()
    else:
        target = None
    
    # Drop columns that exist
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # Identify categorical and numeric columns - FIXED LINE
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Clean numeric columns that may have formatting issues
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = clean_numeric_column(df[col])
    
    # Recalculate after cleaning
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Initialize encoders dictionary if not provided
    if feature_encoders is None:
        feature_encoders = {}
    
    # Encode categorical columns
    for col in categorical_cols:
        if is_training:
            # Training: create new encoder and fit
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = le.fit_transform(df[col])
            feature_encoders[col] = le
        else:
            # Test: reuse existing encoder, handle unseen categories
            if col in feature_encoders:
                le = feature_encoders[col]
                df[col] = df[col].astype(str).fillna('Unknown')
                
                # Handle unseen categories: map to 0 (first class)
                def safe_transform(val):
                    if val in le.classes_:
                        return int(le.transform([val])[0])
                    else:
                        # Return 0 for unseen categories
                        return 0
                
                df[col] = df[col].apply(safe_transform)
            else:
                # Column not in training encoders, fill with 0
                df[col] = 0
    
    # Fill missing values for numeric columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    
    return df, target, feature_encoders


def apply_smote_balancing(X_train, y_train):
    """Apply SMOTE to balance the dataset if imblearn is available."""
    if HAS_IMBLEARN:
        print("  Applying SMOTE to balance classes...")
        # Use SMOTE to oversample minority class
        smote = SMOTE(random_state=42, k_neighbors=5)
        result = smote.fit_resample(X_train, y_train)
        X_balanced, y_balanced = result[0], result[1]
        print(f"  Original class distribution: {np.bincount(y_train)}")
        print(f"  Balanced class distribution: {np.bincount(y_balanced)}")
        return X_balanced, y_balanced
    else:
        print("  Warning: imblearn not installed. Using class_weight instead.")
        print("  Install with: pip install imbalanced-learn")
        return X_train, y_train


def train_model(X_train, y_train):
    """Train an optimized classifier for best accuracy with reasonable resources."""
    
    # Calculate class weights for imbalanced data
    n_samples = len(y_train)
    n_class_0 = np.sum(y_train == 0)
    n_class_1 = np.sum(y_train == 1)
    
    # Use balanced class weights
    scale_pos_weight = n_class_0 / n_class_1 if n_class_1 > 0 else 1.0
    
    # Use LightGBM as primary model - it's faster and memory efficient
    if HAS_LIGHTGBM:
        print("  Training LightGBM (optimized for imbalanced data)...")
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=8,  # Reduced from 12 to prevent overfitting
            learning_rate=0.02,  # Reduced from 0.05 for better generalization
            num_leaves=31,  # Reduced from 64 to prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,  # Increased from 20 to prevent overfitting
            class_weight='balanced',  # Handle imbalance
            reg_alpha=1.0,  # Increased from 0.1 for stronger regularization
            reg_lambda=1.0,  # Increased from 0.1 for stronger regularization
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif HAS_XGBOOST:
        print("  Training XGBoost (optimized for imbalanced data)...")
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,  # Reduced from 10 to prevent overfitting
            learning_rate=0.02,  # Reduced from 0.05 for better generalization
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,  # Increased from 3 to prevent overfitting
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            reg_alpha=1.0,  # Increased from 0.1 for stronger regularization
            reg_lambda=1.0,  # Increased from 0.1 for stronger regularization
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    else:
        print("  Training Gradient Boosting (optimized for imbalanced data)...")
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model


def main():
    print("="*60)
    print("RENEWAL PREDICTION MODEL (Optimized for Imbalanced Data)")
    print("="*60)
    
    print("\n[1/5] Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH, sep=';', low_memory=False)
    print(f"  Training data shape: {train_df.shape}")
    target_counts = train_df['Renew'].value_counts()
    print(f"  Target distribution:")
    for label, count in target_counts.items():
        pct = count / len(train_df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")
    
    print("\n[2/5] Preprocessing training data...")
    X_train, y_train, feature_encoders = preprocess_data(train_df.copy(), is_training=True)
    
    # Encode target variable
    y_train_encoded = (y_train == 'Y').astype(int)
    
    print(f"  Features shape: {X_train.shape}")
    imbalance_ratio = np.sum(y_train_encoded == 0) / np.sum(y_train_encoded == 1)
    print(f"  Class imbalance ratio: 1:{imbalance_ratio:.1f}")
    
    # CORRECTED: First split into train/validation, THEN apply SMOTE
    print("\n[3/5] Splitting data and balancing training set...")
    
    # Split for validation (before SMOTE to avoid data leakage)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.values, y_train_encoded.values, 
        test_size=0.2, random_state=42, stratify=y_train_encoded.values
    )
    
    # Apply SMOTE only to training set (not validation)
    X_tr_balanced, y_tr_balanced = apply_smote_balancing(X_tr, y_tr)
    
    # Train model on balanced training data
    print("\n[4/5] Training and evaluating model...")
    model = train_model(X_tr_balanced, y_tr_balanced)
    
    # Evaluate on validation set with multiple metrics
    val_predictions = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
    
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)
    val_balanced_acc = balanced_accuracy_score(y_val, val_predictions)
    
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")
    print(f"  Validation Balanced Accuracy: {val_balanced_acc:.4f}")
    
    if val_proba is not None:
        val_auc = roc_auc_score(y_val, val_proba)
        print(f"  Validation AUC-ROC: {val_auc:.4f}")
    
    # Train final model on all balanced data
    print("  Training final model on complete balanced dataset...")
    X_train_balanced, y_train_balanced = apply_smote_balancing(X_train.values, y_train_encoded.values)
    final_model = train_model(X_train_balanced, y_train_balanced)
    
    # Load and preprocess test data
    print("\n[5/5] Processing test data and making predictions...")
    test_df = pd.read_csv(TEST_PATH, sep='|', low_memory=False)
    print(f"  Test data shape: {test_df.shape}")
    
    # Preprocess test data using saved encoders from training
    X_test, _, _ = preprocess_data(test_df.copy(), is_training=False, feature_encoders=feature_encoders)
    
    # Ensure test has same features as training
    # Add missing columns with 0
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Remove extra columns
    X_test = X_test[X_train.columns]
    
    print(f"  Test features shape: {X_test.shape}")
    
    # Make predictions
    predictions = final_model.predict(X_test.values)
    predictions = np.asarray(predictions)
    
    # Convert predictions back to Y/N
    prediction_labels = ['Y' if p == 1 else 'N' for p in predictions]
    
    # Save predictions
    print(f"\n  Saving predictions to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for label in prediction_labels:
            f.write(f"{label}\n")
    
    y_count = prediction_labels.count('Y')
    n_count = prediction_labels.count('N')
    total = len(prediction_labels)
    print(f"\n  Prediction distribution:")
    print(f"    Y: {y_count:,} ({y_count/total*100:.1f}%)")
    print(f"    N: {n_count:,} ({n_count/total*100:.1f}%)")
    
    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()