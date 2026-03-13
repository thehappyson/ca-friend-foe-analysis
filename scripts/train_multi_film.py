import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)
import joblib
from typing import Dict, List, Tuple
import argparse

class MultiFilmClassifier:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        self.feature_importance_across_folds = []

    def load_film_data(self, features_csv: str, annotations_csv: str, film_name: str):
        # Load data
        features_df = pd.read_csv(features_csv)
        annotations_df = pd.read_csv(annotations_csv)

        # Merge on frame_path
        merged = features_df.merge(annotations_df, on='frame_path', how='inner')

        # Filter to binary labels (us/them only)
        merged = merged[merged['label'].isin(['us', 'them'])]

        if len(merged) == 0:
            print(f"WARNING: No labeled data found in {film_name}!")
            return None, None, None, film_name

        # Convert labels to binary
        label_map = {'us': 1, 'them': 0}
        y = merged['label'].map(label_map).values

        # Extract features
        feature_cols = [col for col in features_df.columns if col != 'frame_path']
        X = merged[feature_cols].values
        frame_paths = merged['frame_path'].values

        print(f"  {film_name}: {len(X)} samples (Us={np.sum(y==1)}, Them={np.sum(y==0)})")

        return X, y, frame_paths, feature_cols, film_name

    def leave_one_movie_out_cv(self, films_data: List[Tuple], output_dir: str = 'results/multi_film', train_only_data: List[Tuple] = None):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter out None entries
        films_data = [f for f in films_data if f[0] is not None]
        train_only_data = [f for f in (train_only_data or []) if f[0] is not None]

        if len(films_data) < 2:
            raise ValueError("Need at least 2 films for leave-one-out validation!")

        print("\n" + "="*80)
        print("LEAVE-ONE-MOVIE-OUT CROSS-VALIDATION")
        print("="*80)
        print(f"Total LOMO films: {len(films_data)}")
        if train_only_data:
            train_only_names = [f[4] for f in train_only_data]
            print(f"Train-only films (always in training, never tested): {', '.join(train_only_names)}")
        print(f"Strategy: Train on {len(films_data)-1} films, test on 1 held-out film")
        print("="*80 + "\n")

        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []

        # Iterate through each film as test set
        for test_idx in range(len(films_data)):
            test_film = films_data[test_idx]
            train_films = [films_data[i] for i in range(len(films_data)) if i != test_idx]

            X_test, y_test, _, feature_names, test_film_name = test_film

            print(f"\n{'='*80}")
            print(f"FOLD {test_idx + 1}/{len(films_data)}: Testing on {test_film_name}")
            print(f"{'='*80}")

            # Combine training films
            X_train_list = []
            y_train_list = []
            train_film_names = []

            for X_train_film, y_train_film, _, _, film_name in train_films:
                X_train_list.append(X_train_film)
                y_train_list.append(y_train_film)
                train_film_names.append(film_name)

            # Add train-only films to every fold
            for X_to, y_to, _, _, film_name in train_only_data:
                X_train_list.append(X_to)
                y_train_list.append(y_to)
                train_film_names.append(f"{film_name} [train-only]")

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            print(f"Training films: {', '.join(train_film_names)}")
            print(f"Training samples: {len(X_train)} (Us={np.sum(y_train==1)}, Them={np.sum(y_train==0)})")
            print(f"Test samples: {len(X_test)} (Us={np.sum(y_test==1)}, Them={np.sum(y_test==0)})")

            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced'
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            p_per, r_per, f1_per, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

            print(f"\nResults on {test_film_name}:")
            print(f"  Accuracy:          {acc:.3f}  ← misleading with class imbalance")
            print(f"  Balanced Accuracy: {bal_acc:.3f}  ← mean recall across both classes")
            print(f"  Them  precision={p_per[0]:.3f}  recall={r_per[0]:.3f}  f1={f1_per[0]:.3f}")
            print(f"  Us    precision={p_per[1]:.3f}  recall={r_per[1]:.3f}  f1={f1_per[1]:.3f}")

            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Them', 'Us']))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)

            # Store results
            fold_result = {
                'fold': test_idx + 1,
                'test_film': test_film_name,
                'train_films': train_film_names,
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'precision_them': p_per[0],
                'recall_them': r_per[0],
                'f1_them': f1_per[0],
                'precision_us': p_per[1],
                'recall_us': r_per[1],
                'f1_us': f1_per[1],
                'confusion_matrix': cm,
                'n_train': len(X_train),
                'n_test': len(X_test)
            }
            self.results.append(fold_result)

            all_accuracies.append(bal_acc)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

            # Feature importance for this fold
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_,
                'fold': test_idx + 1,
                'test_film': test_film_name
            })
            self.feature_importance_across_folds.append(importance)

            # Save fold-specific model
            fold_dir = output_dir / f'fold_{test_idx + 1}_{test_film_name}'
            fold_dir.mkdir(exist_ok=True)

            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'test_film': test_film_name,
                'train_films': train_film_names
            }
            joblib.dump(model_data, fold_dir / 'model.joblib')

            # Plot confusion matrix for this fold
            self._plot_confusion_matrix(
                cm,
                title=f'Confusion Matrix - Test: {test_film_name}',
                output_path=fold_dir / 'confusion_matrix.png'
            )

        # Summary statistics
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        print(f"Mean Balanced Accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")
        print(f"Mean Precision:         {np.mean(all_precisions):.3f} ± {np.std(all_precisions):.3f}")
        print(f"Mean Recall:            {np.mean(all_recalls):.3f} ± {np.std(all_recalls):.3f}")
        print(f"Mean F1-Score:          {np.mean(all_f1s):.3f} ± {np.std(all_f1s):.3f}")
        print("\nPer-film balanced accuracies:")
        for result in self.results:
            print(f"  {result['test_film']}: {result['balanced_accuracy']:.3f}  (them recall={result['recall_them']:.3f}, us recall={result['recall_us']:.3f})")
        print("="*80 + "\n")

        # Save summary results
        self._save_results(output_dir)

        # Analyze feature importance consistency
        self._analyze_feature_consistency(output_dir)

        summary = {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'mean_precision': np.mean(all_precisions),
            'mean_recall': np.mean(all_recalls),
            'mean_f1': np.mean(all_f1s),
            'per_film_results': self.results
        }

        return summary

    def _plot_confusion_matrix(self, cm, title='Confusion Matrix', output_path=None):

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Them', 'Us'],
                    yticklabels=['Them', 'Us'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _save_results(self, output_dir: Path):

        # Per-fold results
        results_df = pd.DataFrame([
            {
                'fold': r['fold'],
                'test_film': r['test_film'],
                'train_films': ', '.join(r['train_films']),
                'n_train': r['n_train'],
                'n_test': r['n_test'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in self.results
        ])
        results_df.to_csv(output_dir / 'per_fold_results.csv', index=False)
        print(f"✓ Saved per-fold results to {output_dir / 'per_fold_results.csv'}")

        # Summary statistics
        summary_df = pd.DataFrame([{
            'metric': 'accuracy',
            'mean': results_df['accuracy'].mean(),
            'std': results_df['accuracy'].std(),
            'min': results_df['accuracy'].min(),
            'max': results_df['accuracy'].max()
        }, {
            'metric': 'precision',
            'mean': results_df['precision'].mean(),
            'std': results_df['precision'].std(),
            'min': results_df['precision'].min(),
            'max': results_df['precision'].max()
        }, {
            'metric': 'recall',
            'mean': results_df['recall'].mean(),
            'std': results_df['recall'].std(),
            'min': results_df['recall'].min(),
            'max': results_df['recall'].max()
        }, {
            'metric': 'f1',
            'mean': results_df['f1'].mean(),
            'std': results_df['f1'].std(),
            'min': results_df['f1'].min(),
            'max': results_df['f1'].max()
        }])
        summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
        print(f"✓ Saved summary statistics to {output_dir / 'summary_statistics.csv'}")

        # Plot per-film accuracies
        plt.figure(figsize=(10, 6))
        films = [r['test_film'] for r in self.results]
        accs = [r['accuracy'] for r in self.results]
        mean_acc = np.mean(accs)

        bars = plt.bar(films, accs, color='steelblue', edgecolor='black', linewidth=1.5)
        plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_acc:.3f}')
        plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Chance (50%)')

        plt.xlabel('Test Film', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Leave-One-Movie-Out Cross-Validation Results', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.legend(fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_dir / 'per_film_accuracies.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved accuracy plot to {output_dir / 'per_film_accuracies.png'}")

    def _analyze_feature_consistency(self, output_dir: Path):

        # Combine all feature importances
        all_importance = pd.concat(self.feature_importance_across_folds, ignore_index=True)

        # Average importance across folds
        avg_importance = all_importance.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
        avg_importance = avg_importance.sort_values('mean', ascending=False)

        # Save feature importance
        avg_importance.to_csv(output_dir / 'feature_importance_consistency.csv', index=False)
        print(f"✓ Saved feature importance to {output_dir / 'feature_importance_consistency.csv'}")

        # Plot feature importance with error bars
        plt.figure(figsize=(12, 8))
        top_features = avg_importance.head(15)

        plt.barh(range(len(top_features)), top_features['mean'],
                xerr=top_features['std'], capsize=5,
                color='steelblue', edgecolor='black', linewidth=1)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean Importance (across all folds)', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title('Feature Importance Consistency Across Films', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_dir / 'feature_importance_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved feature importance plot to {output_dir / 'feature_importance_consistency.png'}")

        print("\nTop 10 Most Consistent Features:")
        print(avg_importance.head(10).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(
        description='Multi-film leave-one-movie-out cross-validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=''
    )

    parser.add_argument(
        '--films',
        nargs='+',
        required=True,
        help='List of film names (without _features.csv or _annotations.csv suffixes)'
    )
    parser.add_argument(
        '--features-dir',
        default='data/features',
        help='Directory containing feature CSV files'
    )
    parser.add_argument(
        '--annotations-dir',
        default='data/annotated',
        help='Directory containing annotation CSV files'
    )
    parser.add_argument(
        '--output',
        default='results/multi_film',
        help='Output directory for results'
    )
    parser.add_argument(
        '--train-only-films',
        nargs='+',
        default=[],
        help='Films always included in training but never used as test fold (e.g. severely imbalanced films)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MULTI-FILM US VS THEM CLASSIFIER")
    print("Leave-One-Movie-Out Cross-Validation")
    print("="*80)
    print(f"Films: {', '.join(args.films)}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Load all films
    classifier = MultiFilmClassifier(random_state=args.random_state)
    films_data = []

    print("Loading film data...")
    train_only_films_data = []
    for film_name in args.train_only_films:
        features_csv = Path(args.features_dir) / f'{film_name}_features.csv'
        annotations_csv = Path(args.annotations_dir) / f'{film_name}_annotations.csv'
        if not features_csv.exists():
            print(f"WARNING: Features not found for train-only film {film_name}: {features_csv}")
            continue
        if not annotations_csv.exists():
            print(f"WARNING: Annotations not found for train-only film {film_name}: {annotations_csv}")
            continue
        film_data = classifier.load_film_data(str(features_csv), str(annotations_csv), film_name)
        if film_data[0] is not None:
            train_only_films_data.append(film_data)

    for film_name in args.films:
        features_csv = Path(args.features_dir) / f'{film_name}_features.csv'
        annotations_csv = Path(args.annotations_dir) / f'{film_name}_annotations.csv'

        if not features_csv.exists():
            print(f"WARNING: Features not found for {film_name}: {features_csv}")
            continue
        if not annotations_csv.exists():
            print(f"WARNING: Annotations not found for {film_name}: {annotations_csv}")
            continue

        film_data = classifier.load_film_data(str(features_csv), str(annotations_csv), film_name)
        if film_data[0] is not None:
            films_data.append(film_data)

    if len(films_data) < 2:
        print("\nERROR: Need at least 2 films with valid data!")
        return

    # Run leave-one-movie-out CV
    summary = classifier.leave_one_movie_out_cv(films_data, output_dir=args.output, train_only_data=train_only_films_data)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.1%} ± {summary['std_accuracy']:.1%}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
