import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def plot_feature_space(features_csv, annotations_csv, output_dir='results/jud_suess'):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features_df = pd.read_csv(features_csv)
    annotations_df = pd.read_csv(annotations_csv)

    # Merge on frame_path
    merged = features_df.merge(annotations_df, on='frame_path', how='inner')

    # Filter to only 'us' and 'them'
    merged = merged[merged['label'].isin(['us', 'them'])]

    if len(merged) == 0:
        print("No labeled data found!")
        return

    print(f"Loaded {len(merged)} labeled samples")
    print(f"  Us: {len(merged[merged['label'] == 'us'])}")
    print(f"  Them: {len(merged[merged['label'] == 'them'])}")

    # Extract features
    feature_cols = [col for col in features_df.columns if col != 'frame_path']
    X = merged[feature_cols].values
    y = merged['label'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Color mapping
    colors = {'us': '#2E86AB', 'them': '#A23B72'}  # Blue for us, Purple for them
    color_labels = [colors[label] for label in y]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # === PCA ===
    print("\nApplying PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    ax = axes[0]
    for label in ['us', 'them']:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=colors[label], label=label.capitalize(),
                  alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA: "Us" vs "Them" Feature Space', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')

    # Add variance explained text
    total_var = pca.explained_variance_ratio_[:2].sum() * 100
    ax.text(0.02, 0.98, f'Total variance explained: {total_var:.1f}%',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    print(f"PCA explained variance: {pca.explained_variance_ratio_[:2]}")
    print(f"Total variance captured: {total_var:.1f}%")

    # === t-SNE ===
    print("\nApplying t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)

    ax = axes[1]
    for label in ['us', 'them']:
        mask = y == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=colors[label], label=label.capitalize(),
                  alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE: "Us" vs "Them" Feature Space', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    output_path = output_dir / 'feature_space_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    plt.close()

    # === Additional analysis: Feature contributions to PC1 ===
    print("\n" + "="*60)
    print("TOP FEATURES CONTRIBUTING TO PC1 (main axis of separation)")
    print("="*60)

    # Get feature loadings for PC1
    loadings = pd.DataFrame({
        'feature': feature_cols,
        'PC1_loading': pca.components_[0],
        'PC1_loading_abs': np.abs(pca.components_[0])
    }).sort_values('PC1_loading_abs', ascending=False)

    print(loadings.head(10).to_string(index=False))

    # Save loadings
    loadings_path = output_dir / 'pca_loadings.csv'
    loadings.to_csv(loadings_path, index=False)
    print(f"\n✓ Saved PCA loadings to {loadings_path}")

    return X_pca, X_tsne, y

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        features_csv = sys.argv[1]
        annotations_csv = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'results/jud_suess'

        plot_feature_space(features_csv, annotations_csv, output_dir)
    else:
        print("Usage: python visualize_feature_space.py <features.csv> <annotations.csv> [output_dir]")
        print("\nExample:")
        print("  python visualize_feature_space.py data/features/jud_suess_features.csv data/annotated/jud_suess_annotations.csv results/jud_suess")
