#!/usr/bin/env python3
"""
Train a better model that focuses on lighting/composition rather than color.

Since one training video is B&W and one is color, we need to exclude or downweight
color features to learn the actual propaganda visual patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def train_lighting_focused_model(features_csv, annotations_csv, output_dir):
    """
    Train model focusing on lighting and composition, not color.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features_df = pd.read_csv(features_csv)
    annotations_df = pd.read_csv(annotations_csv)

    # Merge
    merged = features_df.merge(annotations_df, on='frame_path')
    merged = merged[merged['label'].isin(['us', 'them'])]

    # Convert labels
    label_map = {'us': 0, 'them': 1}  # Note: swapped to match expected output
    y = merged['label'].map(label_map).values

    # Select only lighting and composition features (exclude color)
    selected_features = [
        'mean_brightness',
        'brightness_std',
        'contrast',
        'low_key_ratio',
        'high_key_ratio',
        'edge_density',
        'center_brightness',
        'vertical_symmetry',
        'horizontal_symmetry',
        'texture_contrast',
        'texture_homogeneity',
        'dark_regions_count',
        'bright_regions_count'
    ]

    # Get only features that exist
    available_features = [f for f in selected_features if f in features_df.columns]

    print(f"Using {len(available_features)} lighting/composition features:")
    for f in available_features:
        print(f"  - {f}")

    X = merged[available_features].values

    print(f"\nPrepared {len(X)} samples")
    print(f"Class distribution: Friend/Us={np.sum(y==0)}, Foe/Them={np.sum(y==1)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\n=== Training Lighting-Focused Model ===")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    y_pred = model.predict(X_test_scaled)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Friend/Us', 'Foe/Them']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Cross-validation
    print("\n=== Cross-Validation ===")
    X_scaled_all = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_scaled_all, y, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Feature Importance ===")
    print(importance_df)

    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': available_features,
        'label_map': label_map
    }

    model_path = output_dir / 'model.joblib'
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")

    # Save feature importance
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance (Lighting-Focused Model)')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Friend/Us', 'Foe/Them'],
                yticklabels=['Friend/Us', 'Foe/Them'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll results saved to {output_dir}")

    return model_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python train_lighting_model.py <features.csv> <annotations.csv> [output_dir]")
        sys.exit(1)

    features_csv = sys.argv[1]
    annotations_csv = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "results/lighting_model"

    train_lighting_focused_model(features_csv, annotations_csv, output_dir)
