import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from src.feature_extraction import VisualFeatureExtractor


def extract_features_from_annotations(annotations_csv, output_csv):
    """
    Extract features from all frames in annotations file.

    Args:
        annotations_csv: Path to annotations CSV
        output_csv: Path to save features CSV
    """
    print("="*70)
    print("EXTRACTING FEATURES FROM ANNOTATED FRAMES")
    print("="*70)

    # Load annotations
    annotations_df = pd.read_csv(annotations_csv)
    print(f"Loaded {len(annotations_df)} annotations")

    # Filter out neutral labels (optional)
    print(f"\nLabel distribution:")
    print(annotations_df['label'].value_counts())

    # Initialize extractor
    extractor = VisualFeatureExtractor()

    # Extract features
    features_list = []
    errors = []

    for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Extracting features"):
        frame_path = row['frame_path']

        try:
            features = extractor.extract_features(frame_path)
            features['frame_path'] = frame_path
            features_list.append(features)
        except Exception as e:
            errors.append((frame_path, str(e)))
            print(f"\nError with {frame_path}: {e}")

    # Save features
    if features_list:
        features_df = pd.DataFrame(features_list)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=False)

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Successfully extracted features: {len(features_list)}")
        print(f"Errors: {len(errors)}")
        print(f"Features per frame: {len(extractor.feature_names)}")
        print(f"\nSaved to: {output_path}")

        if errors:
            print("\nFrames with errors:")
            for path, error in errors[:10]:
                print(f"  {Path(path).name}: {error}")

        print("="*70)
        print("\nYou can now train a model with:")
        print(f"  python train_lighting_model.py {output_csv} {annotations_csv} results/my_model")
    else:
        print("\nNo features extracted!")

    return features_list


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from annotated frames'
    )
    parser.add_argument('annotations', type=str,
                       help='Path to annotations CSV file')
    parser.add_argument('output', type=str,
                       help='Path to save features CSV file')

    args = parser.parse_args()

    extract_features_from_annotations(args.annotations, args.output)


if __name__ == '__main__':
    main()
