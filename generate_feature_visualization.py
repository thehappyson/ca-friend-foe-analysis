#!/usr/bin/env python3
"""
Generate feature visualization for presentation slides.

Creates annotated screenshots showing visual features extracted from video frames.
Useful for explaining the analysis approach in presentations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, List

from src.feature_extraction import VisualFeatureExtractor


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


def extract_sample_frames_from_videos(
    friend_video: str,
    foe_video: str,
    output_dir: str,
    num_samples: int = 3
) -> Tuple[List[Path], List[Path]]:
    """
    Extract sample frames from both friend and foe videos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    friend_frames = []
    foe_frames = []
    
    # Extract from friend video
    print(f"\nExtracting {num_samples} frames from friend video...")
    cap = cv2.VideoCapture(str(friend_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_indices = np.linspace(total_frames * 0.2, total_frames * 0.8, num_samples, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_idx / fps
            output_path = output_dir / f"friend_sample_{i+1}_t{timestamp:.1f}s.jpg"
            cv2.imwrite(str(output_path), frame)
            friend_frames.append(output_path)
            print(f"  Extracted: {output_path.name}")
    
    cap.release()
    
    # Extract from foe video
    print(f"\nExtracting {num_samples} frames from foe video...")
    cap = cv2.VideoCapture(str(foe_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_indices = np.linspace(total_frames * 0.2, total_frames * 0.8, num_samples, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_idx / fps
            output_path = output_dir / f"foe_sample_{i+1}_t{timestamp:.1f}s.jpg"
            cv2.imwrite(str(output_path), frame)
            foe_frames.append(output_path)
            print(f"  Extracted: {output_path.name}")
    
    cap.release()
    
    return friend_frames, foe_frames


def main():
    parser = argparse.ArgumentParser(
        description='Generate feature visualizations for presentation slides'
    )
    parser.add_argument('--friend-video', type=str,
                       default='data/videos/Heimatland.mp4',
                       help='Path to friend video')
    parser.add_argument('--foe-video', type=str,
                       default='data/videos/DerEwigeJude.mp4',
                       help='Path to foe video')
    parser.add_argument('--output-dir', type=str,
                       default='results/presentation_slides',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of sample frames per video')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING FEATURE VISUALIZATIONS FOR PRESENTATION")
    print("="*70)
    
    # Extract sample frames
    frames_dir = output_dir / "sample_frames"
    friend_frames, foe_frames = extract_sample_frames_from_videos(
        args.friend_video,
        args.foe_video,
        frames_dir,
        args.num_samples
    )
    
    print("\n" + "="*70)
    print("GENERATING DETAILED VISUALIZATIONS")
    print("="*70)
    
    # Generate detailed visualizations for each frame
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame_path in enumerate(friend_frames):
        print(f"\nProcessing friend frame {i+1}...")
        output_path = viz_dir / f"friend_detailed_{i+1}.png"
        create_feature_visualization(
            frame_path,
            output_path,
            title=f"Friend Video (Heimatland) - Feature Analysis {i+1}"
        )
    
    for i, frame_path in enumerate(foe_frames):
        print(f"\nProcessing foe frame {i+1}...")
        output_path = viz_dir / f"foe_detailed_{i+1}.png"
        create_feature_visualization(
            frame_path,
            output_path,
            title=f"Foe Video (Der Ewige Jude) - Feature Analysis {i+1}"
        )
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    # Generate comparison visualizations
    for i in range(min(len(friend_frames), len(foe_frames))):
        print(f"\nCreating comparison {i+1}...")
        output_path = viz_dir / f"comparison_{i+1}.png"
        create_comparison_visualization(
            friend_frames[i],
            foe_frames[i],
            output_path
        )
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Sample frames: {frames_dir}")
    print(f"Visualizations: {viz_dir}")
    print("\nGenerated files:")
    print(f"  - {len(friend_frames)} friend frame samples")
    print(f"  - {len(foe_frames)} foe frame samples")
    print(f"  - {len(friend_frames)} friend detailed visualizations")
    print(f"  - {len(foe_frames)} foe detailed visualizations")
    print(f"  - {min(len(friend_frames), len(foe_frames))} comparison visualizations")
    print("\nUse these images in your presentation slides!")


if __name__ == '__main__':
    main()
