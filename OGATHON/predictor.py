#!/usr/bin/env python3
"""
Memory-Efficient Optimized Model for Renewal Prediction
Balanced approach: high accuracy with reasonable memory usage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
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

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "ia_data", "training_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "ia_data", "test_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "predictions.txt")

# Memory optimization mode
USE_LIGHTWEIGHT = True  # Set to False if you have 16GB+ RAM


def clean_numeric_column(series):
    """Clean numeric columns that may have unusual formatting."""
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r'\.(?=\d{3})', '', regex=True)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    return series


def preprocess_data(df, is_training=True, feature_encoders=None):
    """Preprocess the dataframe for model training/prediction."""
    
    columns_to_drop = ['DateAlt', 'KeyMed', 'KeyEnf']
    
    if is_training:
        columns_to_drop.append('Renew')
        target = df['Renew'].copy()
    else:
        target = None
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = clean_numeric_column(df[col])
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if feature_encoders is None:
        feature_encoders = {}
    
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
                
                # Handle unseen categories: map to most frequent class
                def safe_transform(val):
                    if val in le.classes_:
                        return int(le.transform([val])[0])  # type: ignore[index]
                    else:
                        # Return encoded value of first class (most common approach)
                        return 0
                
                df[col] = df[col].apply(safe_transform)
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    
    # Convert to float32 to save memory
    for col in numeric_cols:
        df[col] = df[col].astype('float32')
    
    return df, target, feature_encoders


def train_model(X_train, y_train):
    """Train optimized model based on available resources."""
    
    if USE_LIGHTWEIGHT:
        print("  Using LIGHTWEIGHT mode (memory-efficient)")
        
        # Best single model: LightGBM or XGBoost (very memory efficient)
        if HAS_LIGHTGBM:
            print("  Training optimized LightGBM...")
            model = LGBMClassifier(
                n_estimators=1000,
                max_depth=8,  # Reduced from 12 to prevent overfitting
                learning_rate=0.02,  # Reduced from 0.03 for better generalization
                num_leaves=31,  # Reduced from 50 to prevent overfitting
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=50,  # Increased from 20 to prevent overfitting
                reg_alpha=1.0,  # Increased from 0.1 for stronger regularization
                reg_lambda=1.0,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif HAS_XGBOOST:
            print("  Training optimized XGBoost...")
            scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])
            model = XGBClassifier(
                n_estimators=1000,
                max_depth=8,  # Reduced from 12 to prevent overfitting
                learning_rate=0.02,  # Reduced from 0.03 for better generalization
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,  # Increased from 1, controls sum of weights (not count)
                gamma=0.1,
                reg_alpha=1.0,  # Increased from 0.1 for stronger regularization
                reg_lambda=1.0,
                scale_pos_weight=scale_pos,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method='hist'  # Faster and more memory efficient
            )
        else:
            print("  Training optimized Random Forest...")
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
    else:
        print("  Using FULL POWER mode (requires more memory)")
        
        # Voting ensemble with 3 best models
        estimators = []
        
        if HAS_LIGHTGBM:
            print("    ✓ LightGBM")
            lgbm = LGBMClassifier(
                n_estimators=800,
                max_depth=10,
                learning_rate=0.04,
                num_leaves=40,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            estimators.append(('lgbm', lgbm))
        
        if HAS_XGBOOST:
            print("    ✓ XGBoost")
            scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])
            xgb = XGBClassifier(
                n_estimators=800,
                max_depth=10,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method='hist'
            )
            estimators.append(('xgb', xgb))
        
        print("    ✓ Random Forest")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        estimators.append(('rf', rf))
        
        # Voting ensemble
        model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=1  # Important: don't parallelize the ensemble itself
        )
    
    print("  Fitting model...")
    model.fit(X_train, y_train)
    
    # Clear memory
    gc.collect()
    
    return model


def main():
    print("="*60)
    print("MEMORY-EFFICIENT OPTIMIZED RENEWAL PREDICTION")
    print("="*60)
    
    print(f"\nMode: {'LIGHTWEIGHT' if USE_LIGHTWEIGHT else 'FULL POWER'}")
    print(f"XGBoost: {'✓' if HAS_XGBOOST else '✗'}")
    print(f"LightGBM: {'✓' if HAS_LIGHTGBM else '✗'}")
    
    print("\n[1/6] Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH, sep=';', low_memory=False)
    print(f"  Shape: {train_df.shape}")
    target_counts = train_df['Renew'].value_counts()
    for label, count in target_counts.items():
        pct = count / len(train_df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")
    
    print("\n[2/6] Preprocessing training data...")
    X_train, y_train, feature_encoders = preprocess_data(train_df.copy(), is_training=True)
    y_train_encoded = (y_train == 'Y').astype(int)
    
    print(f"  Features: {X_train.shape}")
    print(f"  Memory usage: {X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Free memory
    del train_df
    gc.collect()
    
    print("\n[3/6] Training model...")
    model = train_model(X_train, y_train_encoded)
    
    print("\n[4/6] Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use fewer jobs for CV to save memory
    cv_scores_acc = cross_val_score(
        model, X_train, y_train_encoded,  # type: ignore[arg-type]
        cv=cv, scoring='accuracy', n_jobs=1, verbose=0
    )
    
    cv_scores_f1 = cross_val_score(
        model, X_train, y_train_encoded,  # type: ignore[arg-type]
        cv=cv, scoring='f1', n_jobs=1, verbose=0
    )
    
    cv_scores_balanced = cross_val_score(
        model, X_train, y_train_encoded,  # type: ignore[arg-type]
        cv=cv, scoring='balanced_accuracy', n_jobs=1, verbose=0
    )
    
    print(f"  Mean Accuracy: {cv_scores_acc.mean():.4f} (+/- {cv_scores_acc.std() * 2:.4f})")
    print(f"  Mean F1-Score: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std() * 2:.4f})")
    print(f"  Mean Balanced Accuracy: {cv_scores_balanced.mean():.4f} (+/- {cv_scores_balanced.std() * 2:.4f})")
    print(f"  Accuracy folds: {[f'{s:.4f}' for s in cv_scores_acc]}")
    print(f"  F1-Score folds: {[f'{s:.4f}' for s in cv_scores_f1]}")
    
    print("\n[5/6] Processing test data...")
    test_df = pd.read_csv(TEST_PATH, sep='|', low_memory=False)
    print(f"  Test shape: {test_df.shape}")
    
    X_test, _, _ = preprocess_data(test_df.copy(), is_training=False, feature_encoders=feature_encoders)
    
    # Align columns
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    del test_df
    gc.collect()
    
    print("\n[6/6] Making predictions...")
    predictions = model.predict(X_test)
    predictions = np.asarray(predictions)
    prediction_labels = ['Y' if p == 1 else 'N' for p in predictions]
    
    with open(OUTPUT_PATH, 'w') as f:
        for label in prediction_labels:
            f.write(f"{label}\n")
    
    print(f"\n✓ Saved to: {OUTPUT_PATH}")
    
    y_count = prediction_labels.count('Y')
    n_count = prediction_labels.count('N')
    total = len(prediction_labels)
    print(f"\nPrediction distribution:")
    print(f"  Y: {y_count:,} ({y_count/total*100:.1f}%)")
    print(f"  N: {n_count:,} ({n_count/total*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✓ COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()