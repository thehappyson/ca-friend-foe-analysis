#!/usr/bin/env python3
"""
Generate feature visualization for presentation slides.

Creates annotated screenshots showing visual features extracted from video frames.
Classifies frames from the SAME video and shows examples of both Friend and Foe
classifications to demonstrate the model's ability to distinguish scenes within one film.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, List, Dict
import joblib

from src.feature_extraction import VisualFeatureExtractor
from src.model import predict_from_features


def create_feature_visualization(
    image_path: str,
    output_path: str,
    title: str = ""
) -> None:
    """
    Create a comprehensive visualization of extracted features.

    Shows:
    - Original frame
    - Brightness/contrast analysis
    - Edge detection
    - Color/saturation analysis
    - Feature values overlay
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract features
    extractor = VisualFeatureExtractor()
    features = extractor.extract_features(image_path)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

    # 1. Original Image
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(img_rgb)
    ax1.set_title('Original Frame', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Brightness Analysis
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(gray, cmap='gray')
    ax2.set_title(f'Brightness Analysis\nMean: {features["mean_brightness"]:.1f}\n'
                  f'Low-key: {features["low_key_ratio"]:.1%} | High-key: {features["high_key_ratio"]:.1%}',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. Brightness Histogram
    ax3 = plt.subplot(2, 4, 3)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax3.plot(hist, color='black', linewidth=2)
    ax3.fill_between(range(256), hist.flatten(), alpha=0.3, color='gray')
    ax3.axvline(x=85, color='blue', linestyle='--', label='Low-key threshold', linewidth=2)
    ax3.axvline(x=170, color='red', linestyle='--', label='High-key threshold', linewidth=2)
    ax3.axvline(x=features["mean_brightness"], color='green', linestyle='-',
                label=f'Mean ({features["mean_brightness"]:.1f})', linewidth=2)
    ax3.set_title('Brightness Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Pixel Intensity')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Edge Detection
    ax4 = plt.subplot(2, 4, 4)
    edges = cv2.Canny(gray, 100, 200)
    ax4.imshow(edges, cmap='gray')
    ax4.set_title(f'Edge Detection\nDensity: {features["edge_density"]:.3f}',
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. Color/Saturation
    ax5 = plt.subplot(2, 4, 5)
    h, s, v = cv2.split(hsv)
    ax5.imshow(s, cmap='jet')
    ax5.set_title(f'Saturation Map\nMean: {features["saturation_mean"]:.1f} | Std: {features["saturation_std"]:.1f}',
                  fontsize=12, fontweight='bold')
    ax5.axis('off')

    # 6. Hue Distribution
    ax6 = plt.subplot(2, 4, 6)
    hue_hist = cv2.calcHist([h], [0], None, [180], [0, 180])
    colors = plt.cm.hsv(np.linspace(0, 1, 180))
    ax6.bar(range(180), hue_hist.flatten(), color=colors, width=1.0)
    ax6.set_title(f'Hue Distribution\nMean: {features["hue_mean"]:.1f}',
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel('Hue Value')
    ax6.set_ylabel('Frequency')
    ax6.grid(alpha=0.3)

    # 7. Composition Analysis
    ax7 = plt.subplot(2, 4, 7)
    # Draw center region
    h_img, w_img = gray.shape
    overlay = img_rgb.copy()
    cv2.rectangle(overlay, (w_img//4, h_img//4), (3*w_img//4, 3*h_img//4), (255, 0, 0), 3)
    ax7.imshow(overlay)
    ax7.set_title(f'Composition\nCenter Brightness: {features["center_brightness"]:.1f}\n'
                  f'V-Symmetry: {features["vertical_symmetry"]:.2f} | H-Symmetry: {features["horizontal_symmetry"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax7.axis('off')

    # 8. Feature Summary Table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

    # Create feature summary
    feature_text = "KEY FEATURES:\n" + "="*40 + "\n\n"

    feature_groups = {
        "Lighting": [
            f"Mean Brightness: {features['mean_brightness']:.1f}",
            f"Contrast: {features['contrast']:.3f}",
            f"Low-key Ratio: {features['low_key_ratio']:.1%}",
            f"High-key Ratio: {features['high_key_ratio']:.1%}"
        ],
        "Color": [
            f"Saturation Mean: {features['saturation_mean']:.1f}",
            f"Saturation Std: {features['saturation_std']:.1f}",
            f"Hue Mean: {features['hue_mean']:.1f}"
        ],
        "Composition": [
            f"Edge Density: {features['edge_density']:.3f}",
            f"Center Brightness: {features['center_brightness']:.1f}",
            f"V-Symmetry: {features['vertical_symmetry']:.2f}",
            f"H-Symmetry: {features['horizontal_symmetry']:.2f}"
        ]
    }

    y_pos = 0.95
    for group_name, group_features in feature_groups.items():
        ax8.text(0.05, y_pos, group_name + ":", fontsize=12, fontweight='bold',
                transform=ax8.transAxes)
        y_pos -= 0.06
        for feat in group_features:
            ax8.text(0.1, y_pos, f"• {feat}", fontsize=10,
                    transform=ax8.transAxes, family='monospace')
            y_pos -= 0.05
        y_pos -= 0.02

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved visualization to {output_path}")


def create_comparison_visualization(
    friend_image: str,
    foe_image: str,
    output_path: str
) -> None:
    """
    Create side-by-side comparison of Friend vs Foe features.
    """
    # Load images
    img_friend = cv2.imread(str(friend_image))
    img_foe = cv2.imread(str(foe_image))

    if img_friend is None or img_foe is None:
        raise ValueError("Could not load images")

    # Convert to RGB
    img_friend_rgb = cv2.cvtColor(img_friend, cv2.COLOR_BGR2RGB)
    img_foe_rgb = cv2.cvtColor(img_foe, cv2.COLOR_BGR2RGB)

    # Extract features
    extractor = VisualFeatureExtractor()
    features_friend = extractor.extract_features(friend_image)
    features_foe = extractor.extract_features(foe_image)

    # Create comparison figure
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Friend vs Foe: Visual Feature Comparison', fontsize=22, fontweight='bold', y=0.98)

    # Friend frame
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img_friend_rgb)
    ax1.set_title('FRIEND Frame Example', fontsize=16, fontweight='bold', color='green')
    ax1.axis('off')

    # Foe frame
    ax2 = plt.subplot(2, 3, 4)
    ax2.imshow(img_foe_rgb)
    ax2.set_title('FOE Frame Example', fontsize=16, fontweight='bold', color='red')
    ax2.axis('off')

    # Brightness comparison
    ax3 = plt.subplot(2, 3, 2)
    categories = ['Mean\nBrightness', 'Contrast', 'Low-key\nRatio']
    friend_vals = [
        features_friend['mean_brightness'] / 255 * 100,
        features_friend['contrast'] * 100,
        features_friend['low_key_ratio'] * 100
    ]
    foe_vals = [
        features_foe['mean_brightness'] / 255 * 100,
        features_foe['contrast'] * 100,
        features_foe['low_key_ratio'] * 100
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax3.bar(x - width/2, friend_vals, width, label='Friend', color='lightgreen', edgecolor='green', linewidth=2)
    ax3.bar(x + width/2, foe_vals, width, label='Foe', color='lightcoral', edgecolor='red', linewidth=2)
    ax3.set_ylabel('Value (%)', fontsize=12)
    ax3.set_title('Lighting Features', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')

    # Color comparison
    ax4 = plt.subplot(2, 3, 5)
    categories = ['Saturation\nMean', 'Saturation\nStd', 'Hue\nMean']
    friend_vals = [
        features_friend['saturation_mean'] / 255 * 100,
        features_friend['saturation_std'] / 255 * 100,
        features_friend['hue_mean'] / 180 * 100
    ]
    foe_vals = [
        features_foe['saturation_mean'] / 255 * 100,
        features_foe['saturation_std'] / 255 * 100,
        features_foe['hue_mean'] / 180 * 100
    ]

    x = np.arange(len(categories))

    ax4.bar(x - width/2, friend_vals, width, label='Friend', color='lightgreen', edgecolor='green', linewidth=2)
    ax4.bar(x + width/2, foe_vals, width, label='Foe', color='lightcoral', edgecolor='red', linewidth=2)
    ax4.set_ylabel('Value (%)', fontsize=12)
    ax4.set_title('Color Features', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')

    # Composition comparison
    ax5 = plt.subplot(2, 3, 3)
    categories = ['Edge\nDensity', 'V-Symmetry', 'H-Symmetry']
    friend_vals = [
        features_friend['edge_density'] * 100,
        features_friend['vertical_symmetry'] * 100,
        features_friend['horizontal_symmetry'] * 100
    ]
    foe_vals = [
        features_foe['edge_density'] * 100,
        features_foe['vertical_symmetry'] * 100,
        features_foe['horizontal_symmetry'] * 100
    ]

    x = np.arange(len(categories))

    ax5.bar(x - width/2, friend_vals, width, label='Friend', color='lightgreen', edgecolor='green', linewidth=2)
    ax5.bar(x + width/2, foe_vals, width, label='Foe', color='lightcoral', edgecolor='red', linewidth=2)
    ax5.set_ylabel('Value (%)', fontsize=12)
    ax5.set_title('Composition Features', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories)
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')

    # Feature difference table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    diff_text = "KEY DIFFERENCES:\n" + "="*50 + "\n\n"

    # Calculate significant differences
    brightness_diff = features_foe['mean_brightness'] - features_friend['mean_brightness']
    lowkey_diff = features_foe['low_key_ratio'] - features_friend['low_key_ratio']
    saturation_diff = features_foe['saturation_mean'] - features_friend['saturation_mean']
    edge_diff = features_foe['edge_density'] - features_friend['edge_density']

    differences = [
        f"Brightness: {brightness_diff:+.1f} ({'darker' if brightness_diff < 0 else 'brighter'} in Foe)",
        f"Low-key Ratio: {lowkey_diff:+.1%} ({'more' if lowkey_diff > 0 else 'less'} dark areas in Foe)",
        f"Saturation: {saturation_diff:+.1f} ({'more' if saturation_diff > 0 else 'less'} saturated in Foe)",
        f"Edge Density: {edge_diff:+.3f} ({'more' if edge_diff > 0 else 'less'} edges in Foe)"
    ]

    y_pos = 0.9
    for diff in differences:
        ax6.text(0.05, y_pos, f"• {diff}", fontsize=11,
                transform=ax6.transAxes, family='monospace')
        y_pos -= 0.12

    # Add interpretation
    y_pos -= 0.05
    ax6.text(0.05, y_pos, "\nTYPICAL PATTERNS:", fontsize=12, fontweight='bold',
            transform=ax6.transAxes)
    y_pos -= 0.12

    patterns = [
        "Friend: Brighter, more saturated colors",
        "Foe: Darker, high contrast, dramatic lighting",
        "Friend: Softer edges, warmer tones",
        "Foe: Sharp edges, cooler/desaturated tones"
    ]

    for pattern in patterns:
        ax6.text(0.05, y_pos, f"  {pattern}", fontsize=10,
                transform=ax6.transAxes, style='italic')
        y_pos -= 0.10

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved comparison visualization to {output_path}")


def classify_and_extract_samples(
    video_path: str,
    model_path: str,
    output_dir: str,
    num_samples: int = 50,
    examples_per_class: int = 3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract frames from ONE video, classify them, and return examples of both classes.

    Returns:
        Tuple of (friend_examples, foe_examples) where each is a list of dicts containing
        frame_path, confidence, timestamp, features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing video: {Path(video_path).name}")
    print(f"Extracting {num_samples} frames for classification...")

    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_indices = np.linspace(total_frames * 0.1, total_frames * 0.9, num_samples, dtype=int)

    extractor = VisualFeatureExtractor()
    frame_data = []

    # Load model to get required features
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict) and 'feature_names' in model_data:
        required_features = model_data['feature_names']
        model_obj = model_data['model']
        scaler_obj = model_data['scaler']
    else:
        # Fallback to all features
        required_features = extractor.feature_names
        model_obj = model_data if not hasattr(model_data, 'model') else model_data.model
        scaler_obj = None if not hasattr(model_data, 'scaler') else model_data.scaler

    print(f"Model requires {len(required_features)} features: {', '.join(required_features[:5])}...")

    # Extract and analyze frames
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = frame_idx / fps
        frame_path = output_dir / f"frame_{frame_idx:06d}_t{timestamp:.1f}s.jpg"
        cv2.imwrite(str(frame_path), frame)

        # Extract features
        features = extractor.extract_features(frame_path)
        # Use only required features
        feature_vector = np.array([[features[name] for name in required_features]])

        # Predict
        prediction = predict_from_features(feature_vector, model_path)[0]

        # Get probability/confidence if model supports it
        try:
            if hasattr(model_obj, 'predict_proba'):
                if scaler_obj is not None:
                    feature_vector_scaled = scaler_obj.transform(feature_vector)
                else:
                    feature_vector_scaled = feature_vector
                proba = model_obj.predict_proba(feature_vector_scaled)[0]
                confidence = proba[prediction]
            else:
                confidence = 1.0  # Default if no probability available
        except:
            confidence = 1.0

        frame_data.append({
            'frame_path': frame_path,
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence,
            'features': features
        })

    cap.release()

    # Separate into friend and foe
    friend_frames = [f for f in frame_data if f['prediction'] == 0]
    foe_frames = [f for f in frame_data if f['prediction'] == 1]

    print(f"\nClassification results:")
    print(f"  Friend frames: {len(friend_frames)} ({len(friend_frames)/len(frame_data)*100:.1f}%)")
    print(f"  Foe frames: {len(foe_frames)} ({len(foe_frames)/len(frame_data)*100:.1f}%)")

    # Select most confident examples
    friend_frames_sorted = sorted(friend_frames, key=lambda x: x['confidence'], reverse=True)
    foe_frames_sorted = sorted(foe_frames, key=lambda x: x['confidence'], reverse=True)

    friend_examples = friend_frames_sorted[:examples_per_class]
    foe_examples = foe_frames_sorted[:examples_per_class]

    print(f"\nSelected {len(friend_examples)} friend examples (highest confidence)")
    print(f"Selected {len(foe_examples)} foe examples (highest confidence)")

    return friend_examples, foe_examples


def main():
    parser = argparse.ArgumentParser(
        description='Generate feature visualizations from ONE video showing both Friend and Foe classifications'
    )
    parser.add_argument('--video', type=str,
                       default='data/videos/judsuess.mp4',
                       help='Path to video file to analyze (judsuess.mp4 contains both classes)')
    parser.add_argument('--model', type=str,
                       default='results/test/model.joblib',
                       help='Path to trained model')
    parser.add_argument('--output-dir', type=str,
                       default='results/presentation_slides',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of frames to sample and classify')
    parser.add_argument('--examples-per-class', type=int, default=3,
                       help='Number of examples to show per class')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GENERATING FEATURE VISUALIZATIONS FOR PRESENTATION")
    print("="*70)
    print(f"Video: {Path(args.video).name}")
    print(f"Model: {Path(args.model).name}")
    print(f"This demonstrates classification of BOTH classes within ONE film")
    print("="*70)

    # Extract and classify frames
    frames_dir = output_dir / "sample_frames"
    friend_examples, foe_examples = classify_and_extract_samples(
        args.video,
        args.model,
        frames_dir,
        args.num_samples,
        args.examples_per_class
    )

    if not friend_examples or not foe_examples:
        print("\nWARNING: Could not find examples of both classes!")
        print(f"Friend examples: {len(friend_examples)}")
        print(f"Foe examples: {len(foe_examples)}")
        if not friend_examples:
            print("\nTry a different video or model - this video may not contain 'Friend' scenes")
        if not foe_examples:
            print("\nTry a different video or model - this video may not contain 'Foe' scenes")
        return

    print("\n" + "="*70)
    print("GENERATING DETAILED VISUALIZATIONS")
    print("="*70)

    # Generate detailed visualizations for each frame
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(args.video).stem

    for i, example in enumerate(friend_examples):
        print(f"\nProcessing FRIEND frame {i+1} (confidence: {example['confidence']:.1%}, t={example['timestamp']:.1f}s)...")
        output_path = viz_dir / f"friend_detailed_{i+1}.png"
        create_feature_visualization(
            example['frame_path'],
            output_path,
            title=f"{video_name} - FRIEND Classification (Confidence: {example['confidence']:.1%}, t={example['timestamp']:.1f}s)"
        )

    for i, example in enumerate(foe_examples):
        print(f"\nProcessing FOE frame {i+1} (confidence: {example['confidence']:.1%}, t={example['timestamp']:.1f}s)...")
        output_path = viz_dir / f"foe_detailed_{i+1}.png"
        create_feature_visualization(
            example['frame_path'],
            output_path,
            title=f"{video_name} - FOE Classification (Confidence: {example['confidence']:.1%}, t={example['timestamp']:.1f}s)"
        )

    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)

    # Generate comparison visualizations (Friend vs Foe from same video!)
    num_comparisons = min(len(friend_examples), len(foe_examples))
    for i in range(num_comparisons):
        print(f"\nCreating comparison {i+1}...")
        output_path = viz_dir / f"comparison_{i+1}.png"
        create_comparison_visualization(
            friend_examples[i]['frame_path'],
            foe_examples[i]['frame_path'],
            output_path
        )

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Sample frames: {frames_dir}")
    print(f"Visualizations: {viz_dir}")
    print(f"\nAnalyzed video: {Path(args.video).name}")
    print(f"\nGenerated files:")
    print(f"  - {len(friend_examples)} FRIEND examples (confidence: {np.mean([e['confidence'] for e in friend_examples]):.1%} avg)")
    print(f"  - {len(foe_examples)} FOE examples (confidence: {np.mean([e['confidence'] for e in foe_examples]):.1%} avg)")
    print(f"  - {len(friend_examples)} friend detailed visualizations")
    print(f"  - {len(foe_examples)} foe detailed visualizations")
    print(f"  - {num_comparisons} comparison visualizations")
    print(f"\n✓ All classifications from the SAME video - demonstrating within-film scene distinction!")
    print("\nUse these images in your presentation slides!")


if __name__ == '__main__':
    main()
