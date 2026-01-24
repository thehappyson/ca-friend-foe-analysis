#!/usr/bin/env python3
"""
Quick test of the entire pipeline using synthetic data.
Run this to verify everything is working before using real video data.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate_test_data import generate_test_dataset
from feature_extraction import extract_features_from_directory
from model import train_and_evaluate_pipeline

def main():
    print("="*60)
    print("Friend-Foe Analysis Pipeline Test")
    print("="*60)

    # Step 1: Generate test data
    print("\n[Step 1/3] Generating synthetic test data...")
    test_dir = Path("data/test_data")
    annotations_file = generate_test_dataset(test_dir, num_us=40, num_them=40)

    # Step 2: Extract features
    print("\n[Step 2/3] Extracting visual features...")
    frames_dir = test_dir / "frames"
    features_file = test_dir / "test_features.csv"
    extract_features_from_directory(frames_dir, features_file)

    # Step 3: Train and evaluate
    print("\n[Step 3/3] Training classifier...")
    results_dir = Path("results/test")
    classifier, results = train_and_evaluate_pipeline(
        features_file,
        annotations_file,
        test_size=0.2,
        output_dir=results_dir
    )

    print("\n" + "="*60)
    print("Pipeline Test Complete!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Model: {results_dir}/model.joblib")
    print(f"  - Feature importance: {results_dir}/feature_importance.csv")
    print(f"  - Plots: {results_dir}/*.png")
    print("\nTest Summary:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1-Score: {results['f1']:.3f}")

    if results['accuracy'] > 0.7:
        print("\n✓ Pipeline is working correctly!")
        print("  You can now use it with real video data.")
    else:
        print("\n⚠ Warning: Low accuracy on test data.")
        print("  This might indicate an issue with the pipeline.")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
