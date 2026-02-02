#!/usr/bin/env python3
"""
Create real training data from actual film videos.

Extracts frames from Friend and Foe films and creates proper annotations.
Based on the film types:
- Heimatland.mp4 = Friend/Us (brighter, positive, community-focused)
- DerEwigeJude.mp4 = Foe/Them (darker, negative, fear-focused)
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

from src.frame_extraction import extract_frames_uniform
from src.feature_extraction import VisualFeatureExtractor


def create_training_dataset(
    friend_video: str,
    foe_video: str,
    output_dir: str,
    frames_per_video: int = 100
):
    """
    Create a balanced training dataset from friend and foe videos.

    Args:
        friend_video: Path to "friend/us" propaganda film
        foe_video: Path to "foe/them" propaganda film
        output_dir: Directory to save frames and annotations
        frames_per_video: Number of frames to extract from each video
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CREATING REAL TRAINING DATASET")
    print("="*70)
    print(f"Friend video: {Path(friend_video).name}")
    print(f"Foe video: {Path(foe_video).name}")
    print(f"Frames per video: {frames_per_video}")
    print()

    annotations = []

    # Extract frames from FRIEND video (labeled as 'us')
    print("\n" + "="*70)
    print("EXTRACTING FRIEND/US FRAMES")
    print("="*70)
    friend_video_path = Path(friend_video)
    friend_output_dir = frames_dir / "us"

    friend_frames = extract_frames_uniform(
        friend_video,
        str(friend_output_dir),
        num_frames=frames_per_video
    )

    for frame_path in friend_frames:
        annotations.append({
            'frame_path': str(frame_path),
            'label': 'us',
            'confidence': 1.0,
            'notes': f'From {friend_video_path.name}',
            'annotator': 'manual'
        })

    print(f"\nExtracted {len(friend_frames)} friend/us frames")

    # Extract frames from FOE video (labeled as 'them')
    print("\n" + "="*70)
    print("EXTRACTING FOE/THEM FRAMES")
    print("="*70)
    foe_video_path = Path(foe_video)
    foe_output_dir = frames_dir / "them"

    foe_frames = extract_frames_uniform(
        foe_video,
        str(foe_output_dir),
        num_frames=frames_per_video
    )

    for frame_path in foe_frames:
        annotations.append({
            'frame_path': str(frame_path),
            'label': 'them',
            'confidence': 1.0,
            'notes': f'From {foe_video_path.name}',
            'annotator': 'manual'
        })

    print(f"\nExtracted {len(foe_frames)} foe/them frames")

    # Save annotations
    annotations_df = pd.DataFrame(annotations)
    annotations_csv = output_dir / "annotations.csv"
    annotations_df.to_csv(annotations_csv, index=False)

    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)

    # Extract features from all frames
    extractor = VisualFeatureExtractor()
    all_features = []

    for annotation in tqdm(annotations, desc="Extracting features"):
        frame_path = annotation['frame_path']
        try:
            features = extractor.extract_features(frame_path)
            features['frame_path'] = frame_path
            all_features.append(features)
        except Exception as e:
            print(f"\nWarning: Could not extract features from {frame_path}: {e}")

    features_df = pd.DataFrame(all_features)
    features_csv = output_dir / "features.csv"
    features_df.to_csv(features_csv, index=False)

    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total frames: {len(annotations)}")
    print(f"Friend/Us frames: {len([a for a in annotations if a['label'] == 'us'])}")
    print(f"Foe/Them frames: {len([a for a in annotations if a['label'] == 'them'])}")
    print(f"\nSaved to: {output_dir}")
    print(f"  Frames: {frames_dir}")
    print(f"  Annotations: {annotations_csv}")
    print(f"  Features: {features_csv}")
    print("\nYou can now train a model with:")
    print(f"  python src/model.py {features_csv} {annotations_csv} results/trained_model")

    return annotations_df, features_df


def main():
    parser = argparse.ArgumentParser(
        description='Create real training dataset from propaganda films'
    )
    parser.add_argument('--friend-video', type=str,
                       default='data/videos/Heimatland.mp4',
                       help='Path to friend/us propaganda film')
    parser.add_argument('--foe-video', type=str,
                       default='data/videos/DerEwigeJude.mp4',
                       help='Path to foe/them propaganda film')
    parser.add_argument('--output-dir', type=str,
                       default='data/training',
                       help='Output directory for training data')
    parser.add_argument('--frames-per-video', type=int, default=100,
                       help='Number of frames to extract from each video')

    args = parser.parse_args()

    create_training_dataset(
        args.friend_video,
        args.foe_video,
        args.output_dir,
        args.frames_per_video
    )


if __name__ == '__main__':
    main()
