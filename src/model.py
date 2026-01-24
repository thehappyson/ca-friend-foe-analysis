import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import joblib


class FriendFoeClassifier:
    """Random Forest classifier for Us vs Them classification."""

    def __init__(self, random_state=42):
        """
        Initialize classifier.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def prepare_data(self, features_csv, annotations_csv):
        """
        Prepare data for training by merging features with annotations.

        Args:
            features_csv: Path to CSV with extracted features
            annotations_csv: Path to CSV with annotations

        Returns:
            X (features), y (labels), frame_paths
        """
        # Load data
        features_df = pd.read_csv(features_csv)
        annotations_df = pd.read_csv(annotations_csv)

        # Merge on frame_path
        merged = features_df.merge(annotations_df, on='frame_path', how='inner')

        # Filter to binary labels (us/them only)
        merged = merged[merged['label'].isin(['us', 'them'])]

        if len(merged) == 0:
            raise ValueError("No labeled data found! Make sure annotations contain 'us' and 'them' labels.")

        # Convert labels to binary
        label_map = {'us': 1, 'them': 0}
        y = merged['label'].map(label_map).values

        # Extract features
        feature_cols = [col for col in features_df.columns if col != 'frame_path']
        X = merged[feature_cols].values

        frame_paths = merged['frame_path'].values

        self.feature_names = feature_cols

        print(f"Prepared {len(X)} samples with {X.shape[1]} features")
        print(f"Class distribution: Us={np.sum(y==1)}, Them={np.sum(y==0)}")

        return X, y, frame_paths

    def train(self, X, y):
        """
        Train the classifier.

        Args:
            X: Feature matrix
            y: Labels (1=us, 0=them)

        Returns:
            Training accuracy
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Compute training accuracy
        train_acc = self.model.score(X_scaled, y)
        print(f"Training accuracy: {train_acc:.3f}")

        return train_acc

    def evaluate(self, X, y, label_names=['Them', 'Us']):
        """
        Evaluate the classifier.

        Args:
            X: Feature matrix
            y: True labels
            label_names: Names for classes

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        # Compute metrics
        acc = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='binary')

        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")

        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=label_names))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds

        Returns:
            Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')

        print(f"\nCross-validation scores ({cv} folds):")
        print(f"Scores: {scores}")
        print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

        return scores

    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from trained model.

        Args:
            top_n: Return top N features (None = all)

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_n:
            df = df.head(top_n)

        return df

    def plot_feature_importance(self, top_n=15, figsize=(10, 6), output_path=None):
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to plot
            figsize: Figure size
            output_path: Optional path to save figure
        """
        importance_df = self.get_feature_importance(top_n=top_n)

        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        plt.show()

    def plot_confusion_matrix(self, cm, label_names=['Them', 'Us'], output_path=None):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            label_names: Class names
            output_path: Optional path to save figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names,
                    yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        plt.show()

    def save_model(self, output_path):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, output_path)
        print(f"Model saved to {output_path}")

    def load_model(self, model_path):
        """Load trained model from disk."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        print(f"Model loaded from {model_path}")


def train_and_evaluate_pipeline(features_csv, annotations_csv, test_size=0.2, output_dir=None):
    """
    Complete pipeline: load data, train, evaluate.

    Args:
        features_csv: Path to features CSV
        annotations_csv: Path to annotations CSV
        test_size: Fraction of data for testing
        output_dir: Directory to save results

    Returns:
        Trained classifier and results
    """
    classifier = FriendFoeClassifier()

    # Prepare data
    X, y, frame_paths = classifier.prepare_data(features_csv, annotations_csv)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train
    print("\n=== Training ===")
    classifier.train(X_train, y_train)

    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    results = classifier.evaluate(X_test, y_test)

    # Cross-validation on full dataset
    print("\n=== Cross-Validation ===")
    cv_scores = classifier.cross_validate(X, y, cv=5)

    # Feature importance
    print("\n=== Feature Importance ===")
    importance_df = classifier.get_feature_importance(top_n=10)
    print(importance_df)

    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        classifier.save_model(output_dir / 'model.joblib')

        # Save feature importance
        importance_df = classifier.get_feature_importance()
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

        # Save plots
        classifier.plot_feature_importance(output_path=output_dir / 'feature_importance.png')
        classifier.plot_confusion_matrix(results['confusion_matrix'],
                                        output_path=output_dir / 'confusion_matrix.png')

        # Save evaluation results
        results_df = pd.DataFrame([{
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }])
        results_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)

        print(f"\nResults saved to {output_dir}")

    return classifier, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        features_csv = sys.argv[1]
        annotations_csv = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "results"

        train_and_evaluate_pipeline(features_csv, annotations_csv, output_dir=output_dir)
    else:
        print("Usage: python model.py <features.csv> <annotations.csv> [output_dir]")
