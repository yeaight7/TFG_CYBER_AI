#!/usr/bin/env python3
"""
Predictive Model for Renewal Approval/Rejection - HACKATHON MODE
Optimized to MAXIMIZE accuracy on imbalanced test data (Y ≈ 12%, N ≈ 88%).
Uses threshold optimization, no SMOTE on final model, keeps predictive columns.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM (preferred) and XGBoost (fallback)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Configuration
RANDOM_STATE = 42
DROP_COLS = ['DateAlt']  # Only drop date column; keep KeyMed/KeyEnf for signal
VAL_SIZE = 0.2

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "ia_data", "training_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "ia_data", "test_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "predictions.txt")


def detect_and_convert_numeric_strings(df, verbose=False):
    """Detect columns that look numeric but are stored as strings (costs with dots/commas)."""
    converted = []
    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(100).astype(str)
        # Check if values look like numbers with thousand separators (dots) or decimal commas
        numeric_pattern = sample.str.match(r'^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+(,\d+)?$')
        if numeric_pattern.mean() > 0.7:  # More than 70% match
            # Convert: remove dots (thousand sep), replace comma with dot (decimal)
            cleaned = df[col].astype(str).str.replace('.', '', regex=False)
            cleaned = cleaned.str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(cleaned, errors='coerce')
            converted.append(col)
    if verbose and converted:
        print(f"  Converted to numeric: {converted}")
    return df


def preprocess_data(df, is_training=True, encoders=None, medians=None, feature_cols=None):
    """
    Preprocess dataframe with OrdinalEncoder for robust handling of unknowns.
    Returns: (X, y, encoders, medians, feature_cols)
    """
    df = df.copy()
    
    # Extract target
    target = None
    if is_training:
        target = (df['Renew'] == 'Y').astype(int)
        df = df.drop(columns=['Renew'])
    
    # Drop configured columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    
    # Detect and convert numeric strings
    df = detect_and_convert_numeric_strings(df, verbose=is_training)
    
    # Identify column types
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if is_training:
        # Fit encoders and compute medians
        encoders = {}
        medians = {}
        
        # Encode categoricals with OrdinalEncoder (handle_unknown=-1)
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('__MISSING__')
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = enc.fit_transform(df[[col]]).ravel()
            encoders[col] = enc
        
        # Fill numeric NaNs with median
        for col in num_cols:
            med = df[col].median() if not df[col].isna().all() else 0
            medians[col] = med
            df[col] = df[col].fillna(med)
        
        feature_cols = df.columns.tolist()
    else:
        # Apply existing encoders
        for col in cat_cols:
            if col in encoders:
                df[col] = df[col].astype(str).fillna('__MISSING__')
                df[col] = encoders[col].transform(df[[col]]).ravel()
            else:
                df[col] = -1  # Unknown column
        
        # Fill numeric NaNs with training medians
        for col in num_cols:
            med = medians.get(col, 0)
            df[col] = df[col].fillna(med)
        
        # Align columns with training
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]
    
    return df, target, encoders, medians, feature_cols


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes accuracy."""
    best_thresh = 0.5
    best_acc = 0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def create_temporal_split(df, y, date_col='DateAlt', val_frac=0.2):
    """Create temporal split based on date if available."""
    if date_col not in df.columns:
        return None, None, None, None, False
    
    try:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        valid = dates.notna()
        if valid.mean() < 0.5:
            return None, None, None, None, False
        
        # Sort by date
        sorted_idx = dates[valid].sort_values().index
        n_val = int(len(sorted_idx) * val_frac)
        train_idx = sorted_idx[:-n_val]
        val_idx = sorted_idx[-n_val:]
        
        return train_idx, val_idx, df.loc[train_idx], df.loc[val_idx], True
    except Exception:
        return None, None, None, None, False


def train_model(X, y, use_class_weight=True):
    """Train the best available model."""
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    scale_pos_weight = n_class_0 / n_class_1 if n_class_1 > 0 else 1.0
    
    if HAS_LIGHTGBM:
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            class_weight='balanced' if use_class_weight else None,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        model_name = "LightGBM"
    elif HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        model_name = "XGBoost"
    else:
        model = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=10,
            learning_rate=0.05,
            min_samples_leaf=30,
            class_weight='balanced' if use_class_weight else None,
            random_state=RANDOM_STATE
        )
        model_name = "HistGradientBoosting"
    
    model.fit(X, y)
    return model, model_name


def main():
    print("=" * 70)
    print("RENEWAL PREDICTION - HACKATHON MODE (Maximize Accuracy)")
    print("=" * 70)
    
    # ===== LOAD DATA =====
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, sep=';', low_memory=False)
    test_df = pd.read_csv(TEST_PATH, sep='|', low_memory=False)
    print(f"  Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # ===== TARGET DISTRIBUTION =====
    target_counts = train_df['Renew'].value_counts()
    print(f"\n  Target distribution in training:")
    for label, count in target_counts.items():
        pct = count / len(train_df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")
    
    # ===== PREPROCESS FULL TRAIN (for final model) =====
    print("\n[2/6] Preprocessing data...")
    X_full, y_full, encoders, medians, feature_cols = preprocess_data(
        train_df.copy(), is_training=True
    )
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  DROP_COLS: {DROP_COLS}")
    
    # ===== VALIDATION SPLIT STRATEGY =====
    print("\n[3/6] Creating validation split and comparing methods...")
    
    # Prepare both split methods and compare
    splits = {}
    
    # Stratified split (always works)
    X_str_tr, X_str_val, y_str_tr, y_str_val = train_test_split(
        X_full.values, y_full.values,
        test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_full.values
    )
    splits['STRATIFIED'] = (X_str_tr, X_str_val, y_str_tr, y_str_val)
    
    # Try temporal split
    train_idx, val_idx, _, _, temporal_ok = create_temporal_split(
        train_df.copy(), y_full, val_frac=VAL_SIZE
    )
    
    if temporal_ok:
        X_temp_tr = X_full.loc[train_idx].values
        X_temp_val = X_full.loc[val_idx].values
        y_temp_tr = y_full.loc[train_idx].values
        y_temp_val = y_full.loc[val_idx].values
        splits['TEMPORAL'] = (X_temp_tr, X_temp_val, y_temp_tr, y_temp_val)
    
    # Quick evaluation of both splits
    print("  Comparing split methods...")
    split_results = {}
    for split_name, (X_tr_s, X_val_s, y_tr_s, y_val_s) in splits.items():
        quick_model, _ = train_model(X_tr_s, y_tr_s, use_class_weight=True)
        proba = quick_model.predict_proba(X_val_s)[:, 1]
        thresh, acc = find_optimal_threshold(y_val_s, proba)
        split_results[split_name] = {'acc': acc, 'thresh': thresh, 'data': (X_tr_s, X_val_s, y_tr_s, y_val_s)}
        print(f"    {split_name}: acc={acc:.4f}, thresh={thresh:.2f}")
    
    # Pick best split
    best_split = max(split_results.keys(), key=lambda k: split_results[k]['acc'])
    X_tr, X_val, y_tr, y_val = split_results[best_split]['data']
    split_method = f"{best_split} (selected - best val accuracy)"
    
    print(f"\n  Selected: {split_method}")
    print(f"  Train: {len(y_tr):,}, Val: {len(y_val):,}")
    print(f"  Val Y rate: {y_val.mean()*100:.1f}%")
    
    # ===== TRAIN VALIDATION MODEL (no SMOTE, use class_weight) =====
    print("\n[4/6] Training validation model...")
    val_model, model_name = train_model(X_tr, y_tr, use_class_weight=True)
    print(f"  Model: {model_name}")
    
    # ===== THRESHOLD OPTIMIZATION =====
    print("\n[5/6] Optimizing threshold for accuracy...")
    val_proba = val_model.predict_proba(X_val)[:, 1]
    
    # Find best threshold
    best_thresh, best_val_acc = find_optimal_threshold(y_val, val_proba)
    
    # Compare with default 0.5
    default_preds = (val_proba >= 0.5).astype(int)
    default_acc = accuracy_score(y_val, default_preds)
    
    print(f"  Default threshold (0.50): Accuracy = {default_acc:.4f}")
    print(f"  Optimal threshold ({best_thresh:.2f}): Accuracy = {best_val_acc:.4f}")
    
    # Additional metrics for info
    opt_preds = (val_proba >= best_thresh).astype(int)
    val_f1 = f1_score(y_val, opt_preds)
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"  (Info) F1-Score: {val_f1:.4f}, AUC: {val_auc:.4f}")
    
    # ===== TRAIN FINAL MODEL ON ALL DATA =====
    print("\n[6/6] Training final model on all training data...")
    final_model, _ = train_model(X_full.values, y_full.values, use_class_weight=True)
    
    # ===== PREPROCESS TEST DATA =====
    print("  Preprocessing test data...")
    X_test, _, _, _, _ = preprocess_data(
        test_df.copy(), is_training=False, 
        encoders=encoders, medians=medians, feature_cols=feature_cols
    )
    
    # ===== MAKE PREDICTIONS WITH OPTIMAL THRESHOLD =====
    test_proba = final_model.predict_proba(X_test.values)[:, 1]
    predictions = (test_proba >= best_thresh).astype(int)
    prediction_labels = ['Y' if p == 1 else 'N' for p in predictions]
    
    # ===== SAVE PREDICTIONS =====
    print(f"\n  Saving predictions to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for label in prediction_labels:
            f.write(f"{label}\n")
    
    # ===== FINAL STATS =====
    y_count = prediction_labels.count('Y')
    n_count = prediction_labels.count('N')
    total = len(prediction_labels)
    y_pct = y_count / total * 100
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_name}")
    print(f"  Split: {split_method}")
    print(f"  Optimal threshold: {best_thresh:.2f}")
    print(f"  Validation accuracy: {best_val_acc:.4f}")
    print(f"\n  Test predictions:")
    print(f"    Y: {y_count:,} ({y_pct:.1f}%)")
    print(f"    N: {n_count:,} ({100-y_pct:.1f}%)")
    
    if y_pct < 5 or y_pct > 20:
        print(f"\n  ⚠️  WARNING: Y prediction rate ({y_pct:.1f}%) seems unusual!")
        print("      Expected range: 5-15% based on training distribution")
    else:
        print(f"\n  ✓ Y prediction rate ({y_pct:.1f}%) is within expected range")
    
    print("\n" + "=" * 70)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()