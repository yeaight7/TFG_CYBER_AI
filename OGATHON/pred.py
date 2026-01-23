#!/usr/bin/env python3
"""
Enhanced Ensemble Renewal Prediction Model
Based on your successful approach with additional ensemble improvements.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "ia_data", "training_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "ia_data", "test_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "predictions.txt")

# Configuration
USE_ENSEMBLE = True  # Set to False for single model (faster)
ENSEMBLE_SIZE = 5  # Number of models in ensemble (2-5 recommended)


def clean_numeric_column(series):
    """Clean numeric columns with unusual formatting."""
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r'\.(?=\d{3})', '', regex=True)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    return series


def preprocess_data(df, is_training=True, feature_encoders=None, numeric_cols_to_convert=None):
    """Preprocess the dataframe for model training/prediction."""
    
    # Track which columns need conversion
    if numeric_cols_to_convert is None:
        numeric_cols_to_convert = []
    
    columns_to_drop = ['DateAlt', 'KeyMed', 'KeyEnf']
    
    if is_training:
        if 'Renew' in df.columns:
            columns_to_drop.append('Renew')
            target = df['Renew'].copy()
        else:
            target = None
    else:
        target = None
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # Identify columns that should be numeric but are object type
    if is_training:
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                converted = clean_numeric_column(df[col])
                if not converted.isna().all():
                    # If conversion worked for most values, keep it
                    non_null_pct = converted.notna().sum() / len(converted)
                    if non_null_pct > 0.5:
                        df[col] = converted
                        numeric_cols_to_convert.append(col)
    else:
        # Apply same conversions as training
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col])
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if feature_encoders is None:
        feature_encoders = {}
    
    # Encode categorical columns
    for col in categorical_cols:
        if is_training:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = le.fit_transform(df[col])
            feature_encoders[col] = le
        else:
            if col in feature_encoders:
                le = feature_encoders[col]
                df[col] = df[col].astype(str).fillna('Unknown')
                
                col_values = df[col].values
                valid_mask = np.isin(col_values, le.classes_)
                
                result = np.zeros(len(col_values), dtype=int)
                if valid_mask.any():
                    result[valid_mask] = le.transform(col_values[valid_mask])
                
                df[col] = result
            else:
                df[col] = 0
    
    # Fill missing values for numeric columns
    for col in numeric_cols:
        median_val = df[col].median() if not df[col].isna().all() else 0
        df[col] = df[col].fillna(median_val)
    
    return df, target, feature_encoders, numeric_cols_to_convert


def find_optimal_threshold(y_true, y_proba, metric='accuracy'):
    """Find optimal classification threshold."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)
        else:
            score = accuracy_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def train_single_model(X_train, y_train, random_seed=42, model_type='lightgbm'):
    """Train a single optimized model."""
    
    if model_type == 'lightgbm' and HAS_LIGHTGBM:
        model = LGBMClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.03,
            num_leaves=40,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=25,
            reg_alpha=0.3,
            reg_lambda=0.3,
            class_weight='balanced',
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1
        )
    elif model_type == 'xgboost' and HAS_XGBOOST:
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        scale_pos = n_class_0 / n_class_1 if n_class_1 > 0 else 1.0
        
        model = XGBClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            scale_pos_weight=scale_pos,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=random_seed,
            n_jobs=-1,
            verbosity=0
        )
    else:
        # Fallback to sklearn GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.85,
            random_state=random_seed
        )
    
    model.fit(X_train, y_train)
    return model


def train_ensemble(X_train, y_train, n_models=3):
    """Train an ensemble of models with different seeds."""
    models = []
    seeds = [42, 123, 456, 789, 2024][:n_models]
    
    print(f"  Training ensemble of {n_models} models...")
    
    for i, seed in enumerate(seeds, 1):
        print(f"    Model {i}/{n_models} (seed={seed})...", end=' ')
        
        # Alternate between LightGBM and XGBoost if both available
        if HAS_LIGHTGBM and HAS_XGBOOST:
            model_type = 'lightgbm' if i % 2 == 1 else 'xgboost'
        elif HAS_LIGHTGBM:
            model_type = 'lightgbm'
        elif HAS_XGBOOST:
            model_type = 'xgboost'
        else:
            model_type = 'sklearn'
        
        model = train_single_model(X_train, y_train, random_seed=seed, model_type=model_type)
        models.append(model)
        print(f"✓ ({model_type})")
    
    return models


def predict_ensemble(models, X):
    """Make predictions using ensemble of models."""
    probas = []
    
    for model in models:
        proba = model.predict_proba(X)[:, 1]
        probas.append(proba)
    
    # Average probabilities
    avg_proba = np.mean(probas, axis=0)
    return avg_proba


def main():
    print("="*70)
    print("ENHANCED ENSEMBLE RENEWAL PREDICTION MODEL")
    print("="*70)
    
    ensemble_status = f"ENSEMBLE ({ENSEMBLE_SIZE} models)" if USE_ENSEMBLE else "SINGLE MODEL"
    print(f"\nMode: {ensemble_status}")
    print(f"LightGBM: {'✓' if HAS_LIGHTGBM else '✗'}")
    print(f"XGBoost: {'✓' if HAS_XGBOOST else '✗'}")
    
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, sep=';', low_memory=False)
    test_df = pd.read_csv(TEST_PATH, sep='|', low_memory=False)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")
    
    target_counts = train_df['Renew'].value_counts()
    for label, count in target_counts.items():
        pct = count / len(train_df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")
    
    print("\n[2/6] Preprocessing training data...")
    X_train, y_train, feature_encoders, numeric_conversions = preprocess_data(
        train_df.copy(), is_training=True
    )
    y_train_encoded = (y_train == 'Y').astype(int)
    
    print(f"  Features: {X_train.shape[1]} columns")
    print(f"  Converted to numeric: {numeric_conversions}")
    
    print("\n[3/6] Creating validation split...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.values, y_train_encoded.values,
        test_size=0.2, random_state=42, stratify=y_train_encoded.values
    )
    print(f"  Train: {len(X_tr):,}, Val: {len(X_val):,}")
    print(f"  Val Y rate: {np.mean(y_val)*100:.1f}%")
    
    print("\n[4/6] Training model(s)...")
    if USE_ENSEMBLE:
        models = train_ensemble(X_tr, y_tr, n_models=ENSEMBLE_SIZE)
        val_proba = predict_ensemble(models, X_val)
    else:
        print("  Training single LightGBM model...")
        model = train_single_model(X_tr, y_tr)
        val_proba = model.predict_proba(X_val)[:, 1]
        models = [model]  # Wrap in list for consistency
    
    print("\n[5/6] Optimizing threshold...")
    optimal_threshold, optimal_accuracy = find_optimal_threshold(y_val, val_proba)
    
    val_pred_opt = (val_proba >= optimal_threshold).astype(int)
    val_f1 = f1_score(y_val, val_pred_opt)
    val_auc = roc_auc_score(y_val, val_proba)
    
    print(f"  Optimal threshold: {optimal_threshold:.2f}")
    print(f"  Validation accuracy: {optimal_accuracy:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")
    print(f"  AUC-ROC: {val_auc:.4f}")
    
    # Compare with default threshold
    val_pred_default = (val_proba >= 0.5).astype(int)
    default_accuracy = accuracy_score(y_val, val_pred_default)
    print(f"  Improvement over default (0.50): {optimal_accuracy - default_accuracy:+.4f}")
    
    print("\n[6/6] Training final model(s) and predicting...")
    if USE_ENSEMBLE:
        final_models = train_ensemble(X_train.values, y_train_encoded.values, n_models=ENSEMBLE_SIZE)
    else:
        print("  Training final model on all training data...")
        final_model = train_single_model(X_train.values, y_train_encoded.values)
        final_models = [final_model]
    
    # Preprocess test data
    print("  Preprocessing test data...")
    X_test, _, _, _ = preprocess_data(
        test_df.copy(), 
        is_training=False, 
        feature_encoders=feature_encoders,
        numeric_cols_to_convert=numeric_conversions
    )
    
    # Align columns
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    # Make predictions
    if USE_ENSEMBLE:
        test_proba = predict_ensemble(final_models, X_test.values)
    else:
        test_proba = final_models[0].predict_proba(X_test.values)[:, 1]
    
    predictions = (test_proba >= optimal_threshold).astype(int)
    prediction_labels = ['Y' if p == 1 else 'N' for p in predictions]
    
    # Save predictions
    with open(OUTPUT_PATH, 'w') as f:
        for label in prediction_labels:
            f.write(f"{label}\n")
    
    print(f"  Saved to: {OUTPUT_PATH}")
    
    y_count = prediction_labels.count('Y')
    n_count = prediction_labels.count('N')
    total = len(prediction_labels)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"  Model: {ensemble_status}")
    print(f"  Optimal threshold: {optimal_threshold:.2f}")
    print(f"  Validation accuracy: {optimal_accuracy:.4f}")
    print(f"  Test predictions:")
    print(f"    Y: {y_count:,} ({y_count/total*100:.1f}%)")
    print(f"    N: {n_count:,} ({n_count/total*100:.1f}%)")
    
    expected_y_rate = np.mean(y_train_encoded) * 100
    actual_y_rate = y_count / total * 100
    if abs(actual_y_rate - expected_y_rate) < 5:
        print(f"  ✓ Y prediction rate ({actual_y_rate:.1f}%) is within expected range")
    else:
        print(f"  ⚠ Y prediction rate ({actual_y_rate:.1f}%) differs from training ({expected_y_rate:.1f}%)")
    
    print("="*70)
    print("✓ COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()