#!/usr/bin/env python3
"""
Optimized Predictive Model for Renewal Approval/Rejection
This script trains an ensemble of optimized classifiers for maximum accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM for better performance
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "ia_data", "training_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "ia_data", "test_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "predicciones.txt")


def clean_numeric_column(series):
    """Clean numeric columns that may have unusual formatting."""
    if series.dtype == object:
        # Replace dots used as thousand separators and handle decimal commas
        cleaned = series.astype(str).str.replace(r'\.(?=\d{3})', '', regex=True)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    return series


def preprocess_data(df, is_training=True, feature_encoders=None):
    """Preprocess the dataframe for model training/prediction."""
    
    # Drop non-predictive columns
    columns_to_drop = ['DateAlt', 'KeyMed', 'KeyEnf']
    
    if is_training:
        columns_to_drop.append('Renew')
        target = df['Renew'].copy()
    else:
        target = None
    
    # Drop columns that exist
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Clean numeric columns that may have formatting issues
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = clean_numeric_column(df[col])
    
    # Recalculate after cleaning
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical columns
    if feature_encoders is None:
        feature_encoders = {}
    
    for col in categorical_cols:
        if is_training:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = le.fit_transform(df[col])
            feature_encoders[col] = le
        else:
            # Use existing encoder
            if col in feature_encoders:
                df[col] = df[col].astype(str).fillna('Unknown')
                # Handle unseen categories
                le = feature_encoders[col]
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                df[col] = le.transform(df[col])
    
    # Fill missing values for numeric columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    
    return df, target, feature_encoders


def train_optimized_ensemble(X_train, y_train):
    """Train an optimized stacking ensemble for maximum accuracy."""
    
    print("  Building optimized base models...")
    
    # Base models with fine-tuned hyperparameters
    base_models = []
    
    # Random Forest - Optimized for high accuracy
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',  # Handle imbalance
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('rf', rf))
    
    # Extra Trees - More randomness, good for ensemble diversity
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=43,
        n_jobs=-1
    )
    base_models.append(('et', et))
    
    # Gradient Boosting - Sequential learning
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    base_models.append(('gb', gb))
    
    # XGBoost - If available, excellent performance
    if HAS_XGBOOST:
        print("  ✓ Using XGBoost")
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        base_models.append(('xgb', xgb))
    
    # LightGBM - If available, fast and accurate
    if HAS_LIGHTGBM:
        print("  ✓ Using LightGBM")
        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        base_models.append(('lgbm', lgbm))
    
    # Meta-learner: Logistic Regression to combine base models
    meta_learner = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    # Create stacking ensemble
    print("  Creating stacking ensemble...")
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    print("  Training ensemble (this may take a few minutes)...")
    stacking_model.fit(X_train, y_train)
    
    return stacking_model


def main():
    print("="*60)
    print("OPTIMIZED RENEWAL PREDICTION MODEL")
    print("="*60)
    
    print("\n[1/6] Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH, sep=';', low_memory=False)
    print(f"  Shape: {train_df.shape}")
    print(f"  Target distribution:")
    target_counts = train_df['Renew'].value_counts()
    for label, count in target_counts.items():
        pct = count / len(train_df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")
    
    print("\n[2/6] Preprocessing training data...")
    X_train, y_train, feature_encoders = preprocess_data(train_df.copy(), is_training=True)
    
    # Encode target variable
    y_train_encoded = (y_train == 'Y').astype(int)
    
    print(f"  Features shape: {X_train.shape}")
    print(f"  Class balance ratio: 1:{len(y_train_encoded[y_train_encoded==0]) / len(y_train_encoded[y_train_encoded==1]):.1f}")
    
    print("\n[3/6] Training optimized ensemble model...")
    model = train_optimized_ensemble(X_train, y_train_encoded)
    
    print("\n[4/6] Evaluating with stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train_encoded, 
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    print(f"  Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Individual folds: {[f'{s:.4f}' for s in cv_scores]}")
    
    print("\n[5/6] Processing test data...")
    test_df = pd.read_csv(TEST_PATH, sep='|', low_memory=False)
    print(f"  Test shape: {test_df.shape}")
    
    # Preprocess test data with same encoders
    X_test, _, _ = preprocess_data(test_df.copy(), is_training=False, feature_encoders=feature_encoders)
    
    # Ensure test has same features as training
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    
    X_test = X_test[X_train.columns]
    print(f"  Aligned test shape: {X_test.shape}")
    
    print("\n[6/6] Making predictions...")
    predictions = model.predict(X_test)
    prediction_labels = ['Y' if p == 1 else 'N' for p in predictions.tolist()]
    
    # Save predictions
    with open(OUTPUT_PATH, 'w') as f:
        for label in prediction_labels:
            f.write(f"{label}\n")
    
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"\n  Prediction distribution:")
    y_count = prediction_labels.count('Y')
    n_count = prediction_labels.count('N')
    total = len(prediction_labels)
    print(f"    Y: {y_count:,} ({y_count/total*100:.1f}%)")
    print(f"    N: {n_count:,} ({n_count/total*100:.1f}%)")
    
    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()