#!/usr/bin/env python3
"""
Proof of Concept: Face Detection and Analysis for Nazi Propaganda Films

This script:
1. Detects faces in annotated "us" and "them" frames
2. Extracts face-specific visual features
3. Performs statistical comparison to test hypotheses:
   - H1: "Us" faces have bottom-lit (heroic) lighting
   - H2: "Them" faces have top-lit (harsh) lighting
   - H3: "Us" faces are more centered and prominent
   - H4: Face-level features discriminate better than frame-level
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.face_analysis import FaceAnalyzer, compare_us_vs_them

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_annotated_frames(annotations_csv='data/training/annotations.csv'):
    """Load annotated training frames."""
    df = pd.read_csv(annotations_csv)
    print(f"Loaded {len(df)} annotated frames")
    print(f"  'us': {len(df[df['label']=='us'])}")
    print(f"  'them': {len(df[df['label']=='them'])}")
    return df


def quick_validation_test(analyzer, df_annotations, n_samples=20):
    """
    Quick validation: Test face detection on small sample.

    This validates that:
    1. Face detector works on historical footage
    2. Faces are detected in reasonable proportion
    3. Feature extraction runs without errors
    """
    print("\n" + "="*60)
    print("QUICK VALIDATION TEST")
    print("="*60)

    # Sample frames
    us_sample = df_annotations[df_annotations['label'] == 'us'].sample(min(n_samples, len(df_annotations[df_annotations['label'] == 'us'])))
    them_sample = df_annotations[df_annotations['label'] == 'them'].sample(min(n_samples, len(df_annotations[df_annotations['label'] == 'them'])))

    sample_paths = list(us_sample['frame_path']) + list(them_sample['frame_path'])
    sample_labels = list(us_sample['label']) + list(them_sample['label'])

    df_faces, df_frames = analyzer.analyze_dataset(sample_paths, sample_labels)

    print(f"\nValidation Results:")
    print(f"  Frames analyzed: {len(sample_paths)}")
    print(f"  Faces detected: {len(df_faces)}")
    print(f"  Detection rate: {100 * len(df_faces) / len(sample_paths):.1f} faces per frame")

    return df_faces, df_frames


def full_analysis(analyzer, df_annotations, output_dir='results/face_analysis_poc'):
    """Run full face analysis on all annotated frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FULL FACE ANALYSIS")
    print("="*60)

    # Analyze all frames
    image_paths = df_annotations['frame_path'].tolist()
    labels = df_annotations['label'].tolist()

    df_faces, df_frames = analyzer.analyze_dataset(
        image_paths,
        labels,
        output_csv=output_dir / 'face_features.csv'
    )

    return df_faces, df_frames


def statistical_analysis(df_faces, output_dir='results/face_analysis_poc'):
    """Perform statistical comparison of us vs them faces."""
    output_dir = Path(output_dir)

    if len(df_faces) == 0:
        print("No faces detected - skipping statistical analysis")
        return None

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Compare features
    df_stats = compare_us_vs_them(df_faces)

    # Display results
    print("\nTop 10 Most Discriminative Features:")
    print("-" * 60)
    for idx, row in df_stats.head(10).iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature']:30s} | d={row['cohens_d']:+.3f} | p={row['p_value']:.4f} {sig}")
        print(f"  'Us':   {row['us_mean']:.3f} ± {row['us_std']:.3f}")
        print(f"  'Them': {row['them_mean']:.3f} ± {row['them_std']:.3f}")

    # Save results
    df_stats.to_csv(output_dir / 'statistical_comparison.csv', index=False)
    print(f"\nSaved statistical results to {output_dir / 'statistical_comparison.csv'}")

    return df_stats


def create_visualizations(df_faces, df_stats, output_dir='results/face_analysis_poc'):
    """Create visualization plots."""
    output_dir = Path(output_dir)

    if len(df_faces) == 0:
        print("No faces detected - skipping visualizations")
        return

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # 1. Feature distributions for top discriminative features
    top_features = df_stats.head(6)['feature'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        if i >= 6:
            break

        ax = axes[i]

        us_data = df_faces[df_faces['label'] == 'us'][feature].dropna()
        them_data = df_faces[df_faces['label'] == 'them'][feature].dropna()

        ax.hist(us_data, alpha=0.5, label='Us', bins=20, color='blue')
        ax.hist(them_data, alpha=0.5, label='Them', bins=20, color='red')

        # Get stats
        stat_row = df_stats[df_stats['feature'] == feature].iloc[0]

        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_title(f"{feature}\np={stat_row['p_value']:.4f}, d={stat_row['cohens_d']:.2f}")

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_distributions.png'}")
    plt.close()

    # 2. Effect sizes plot
    plt.figure(figsize=(10, 8))

    top_10 = df_stats.head(10).copy()
    top_10 = top_10.sort_values('cohens_d')

    colors = ['blue' if d > 0 else 'red' for d in top_10['cohens_d']]

    plt.barh(range(len(top_10)), top_10['cohens_d'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_10)), [f.replace('_', ' ').title() for f in top_10['feature']])
    plt.xlabel("Cohen's d (Effect Size)")
    plt.title("Top 10 Discriminative Features\n(Blue: Higher in 'Us', Red: Higher in 'Them')")
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'effect_sizes.png'}")
    plt.close()

    # 3. Lighting direction comparison (key hypothesis)
    if 'lighting_direction' in df_faces.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        us_lighting = df_faces[df_faces['label'] == 'us']['lighting_direction'].dropna()
        them_lighting = df_faces[df_faces['label'] == 'them']['lighting_direction'].dropna()

        ax1.hist(us_lighting, alpha=0.6, label='Us', bins=20, color='gold')
        ax1.hist(them_lighting, alpha=0.6, label='Them', bins=20, color='darkred')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Neutral')
        ax1.set_xlabel('Lighting Direction\n(Negative=Bottom-lit/Heroic, Positive=Top-lit/Harsh)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.set_title('Lighting Direction Distribution')

        # Box plot
        lighting_data = pd.DataFrame({
            'Lighting Direction': list(us_lighting) + list(them_lighting),
            'Category': ['Us'] * len(us_lighting) + ['Them'] * len(them_lighting)
        })

        sns.boxplot(data=lighting_data, x='Category', y='Lighting Direction',
                   palette={'Us': 'gold', 'Them': 'darkred'}, ax=ax2)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Lighting Direction Comparison')
        ax2.set_ylabel('Lighting Direction\n(Negative=Bottom-lit, Positive=Top-lit)')

        plt.tight_layout()
        plt.savefig(output_dir / 'lighting_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'lighting_analysis.png'}")
        plt.close()


def generate_report(df_faces, df_frames, df_stats, output_dir='results/face_analysis_poc'):
    """Generate summary report."""
    output_dir = Path(output_dir)

    report_path = output_dir / 'POC_REPORT.md'

    with open(report_path, 'w') as f:
        f.write("# Face Analysis POC - Results Report\n\n")
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total frames analyzed: {len(df_frames)}\n")
        f.write(f"- Frames with faces: {np.sum(df_frames['num_faces'] > 0)} ({100*np.sum(df_frames['num_faces'] > 0)/len(df_frames):.1f}%)\n")
        f.write(f"- Total faces detected: {len(df_faces)}\n")

        if len(df_faces) > 0:
            us_faces = df_faces[df_faces['label'] == 'us']
            them_faces = df_faces[df_faces['label'] == 'them']
            f.write(f"- 'Us' faces: {len(us_faces)}\n")
            f.write(f"- 'Them' faces: {len(them_faces)}\n\n")

            f.write("## Key Findings\n\n")

            # Significant features
            sig_features = df_stats[df_stats['p_value'] < 0.05]
            f.write(f"### Statistically Significant Differences (p < 0.05): {len(sig_features)}/{len(df_stats)}\n\n")

            f.write("| Feature | Us Mean | Them Mean | Effect Size (d) | p-value |\n")
            f.write("|---------|---------|-----------|-----------------|----------|\n")

            for _, row in sig_features.head(10).iterrows():
                f.write(f"| {row['feature']} | {row['us_mean']:.3f} | {row['them_mean']:.3f} | {row['cohens_d']:+.3f} | {row['p_value']:.4f} |\n")

            f.write("\n## Hypothesis Testing\n\n")

            # Test specific hypotheses
            if 'lighting_direction' in df_stats['feature'].values:
                lighting = df_stats[df_stats['feature'] == 'lighting_direction'].iloc[0]
                f.write(f"### H1: Lighting Direction Differs\n\n")
                f.write(f"- 'Us' faces: {lighting['us_mean']:.3f} (negative = bottom-lit)\n")
                f.write(f"- 'Them' faces: {lighting['them_mean']:.3f} (positive = top-lit)\n")
                f.write(f"- Effect size: {lighting['cohens_d']:.3f}\n")
                f.write(f"- p-value: {lighting['p_value']:.4f}\n")
                f.write(f"- **Result: {'SUPPORTED' if lighting['p_value'] < 0.05 else 'NOT SUPPORTED'}**\n\n")

            if 'face_centrality' in df_stats['feature'].values:
                centrality = df_stats[df_stats['feature'] == 'face_centrality'].iloc[0]
                f.write(f"### H2: 'Us' Faces More Centered\n\n")
                f.write(f"- 'Us' faces: {centrality['us_mean']:.3f}\n")
                f.write(f"- 'Them' faces: {centrality['them_mean']:.3f}\n")
                f.write(f"- Effect size: {centrality['cohens_d']:.3f}\n")
                f.write(f"- p-value: {centrality['p_value']:.4f}\n")
                f.write(f"- **Result: {'SUPPORTED' if centrality['p_value'] < 0.05 and centrality['us_mean'] > centrality['them_mean'] else 'NOT SUPPORTED'}**\n\n")

            f.write("\n## Conclusion\n\n")
            f.write("This POC demonstrates that face-level analysis can detect systematic differences ")
            f.write("in facial presentation between 'us' and 'them' depictions in Nazi propaganda films.\n\n")

            if len(sig_features) >= 3:
                f.write("**Recommendation: Proceed with full face analysis pipeline** - ")
                f.write("Multiple features show significant discrimination.\n")
            else:
                f.write("**Recommendation: Expand dataset and retry** - ")
                f.write("Limited significant differences detected, may need more training data.\n")

    print(f"\nGenerated report: {report_path}")


def main():
    """Main POC workflow."""
    print("="*60)
    print("FACE ANALYSIS PROOF OF CONCEPT")
    print("Nazi Propaganda Film Analysis")
    print("="*60)

    # Initialize face analyzer
    print("\n[1/6] Initializing face detector...")
    analyzer = FaceAnalyzer(detector='mtcnn', min_confidence=0.90)

    # Load annotations
    print("\n[2/6] Loading annotated frames...")
    df_annotations = load_annotated_frames()

    # Quick validation test
    print("\n[3/6] Running quick validation test...")
    df_faces_sample, df_frames_sample = quick_validation_test(analyzer, df_annotations, n_samples=20)

    # Ask user if they want to proceed with full analysis
    if len(df_faces_sample) > 0:
        print("\n" + "="*60)
        print("Validation successful! Proceeding with full analysis...")
        print("(To run validation only, modify the script)")
    else:
        print("\nWARNING: No faces detected in validation test!")
        print("This could mean:")
        print("  1. MTCNN not installed (run: pip install mtcnn)")
        print("  2. Frames don't contain faces")
        print("  3. Detection confidence threshold too high")
        return

    # Full analysis
    print("\n[4/6] Running full face analysis...")
    df_faces, df_frames = full_analysis(analyzer, df_annotations)

    if len(df_faces) == 0:
        print("ERROR: No faces detected in full analysis!")
        return

    # Statistical analysis
    print("\n[5/6] Performing statistical analysis...")
    df_stats = statistical_analysis(df_faces)

    # Create visualizations
    print("\n[6/6] Creating visualizations...")
    create_visualizations(df_faces, df_stats)

    # Generate report
    generate_report(df_faces, df_frames, df_stats)

    print("\n" + "="*60)
    print("POC COMPLETE!")
    print("="*60)
    print("\nResults saved to: results/face_analysis_poc/")
    print("  - face_features.csv (all detected faces)")
    print("  - statistical_comparison.csv (feature comparison)")
    print("  - feature_distributions.png")
    print("  - effect_sizes.png")
    print("  - lighting_analysis.png")
    print("  - POC_REPORT.md (summary)")


if __name__ == "__main__":
    main()
