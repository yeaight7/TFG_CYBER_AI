#!/usr/bin/env python3
"""
Predictive Model for Renewal Approval/Rejection
This script trains a classification model to predict if a renewal should be approved (Y) or rejected (N).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

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


def preprocess_data(df, is_training=True):
    """Preprocess the dataframe for model training/prediction."""
    
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
    
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Fill missing values for numeric columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    
    return df, target


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    # Use Random Forest for better handling of mixed data types
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH, sep=';', low_memory=False)
    print(f"Training data shape: {train_df.shape}")
    print(f"Target distribution:\n{train_df['Renew'].value_counts()}")
    
    print("\nPreprocessing training data...")
    X_train, y_train = preprocess_data(train_df.copy(), is_training=True)
    
    # Encode target variable
    y_train_encoded = (y_train == 'Y').astype(int)
    
    print(f"Features shape: {X_train.shape}")
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train_encoded)
    
    # Cross-validation on training data
    print("\nEvaluating model with cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Load and preprocess test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_PATH, sep='|', low_memory=False)
    print(f"Test data shape: {test_df.shape}")
    
    print("Preprocessing test data...")
    # Need to ensure same columns
    # Get common columns
    train_columns = set(train_df.columns) - {'Renew', 'DateAlt', 'KeyMed', 'KeyEnf'}
    test_columns = set(test_df.columns) - {'DateAlt', 'KeyMed', 'KeyEnf'}
    
    # Preprocess test data
    X_test, _ = preprocess_data(test_df.copy(), is_training=False)
    
    # Ensure test has same features as training
    # Add missing columns with 0
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Remove extra columns
    X_test = X_test[X_train.columns]
    
    print(f"Test features shape: {X_test.shape}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Convert predictions back to Y/N
    prediction_labels = ['Y' if p == 1 else 'N' for p in predictions]
    
    # Save predictions
    print(f"\nSaving predictions to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for label in prediction_labels:
            f.write(f"{label}\n")
    
    print(f"\nPrediction distribution:")
    print(f"  Y: {prediction_labels.count('Y')}")
    print(f"  N: {prediction_labels.count('N')}")
    print(f"\nTotal predictions: {len(prediction_labels)}")
    print("Done!")


if __name__ == "__main__":
    main()