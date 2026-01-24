"""
Generate synthetic test data for testing the pipeline without actual video files.
Creates fake frames with distinct visual characteristics for "us" vs "them".
"""
import cv2
import numpy as np
from pathlib import Path
import pandas as pd


def generate_us_frame(width=640, height=480):
    """
    Generate a synthetic "us" (ingroup) frame.
    Characteristics: bright, high-key lighting, symmetrical, warm colors.
    """
    # Start with bright background
    img = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Add warm color cast (slight yellow/orange tint)
    img[:, :, 0] = np.clip(img[:, :, 0] + 20, 0, 255)  # Blue channel
    img[:, :, 1] = np.clip(img[:, :, 1] + 30, 0, 255)  # Green channel
    img[:, :, 2] = np.clip(img[:, :, 2] + 40, 0, 255)  # Red channel

    # Add symmetrical geometric shapes (order, power)
    center_x, center_y = width // 2, height // 2

    # Symmetrical rectangles
    cv2.rectangle(img, (center_x - 150, center_y - 100),
                  (center_x + 150, center_y + 100), (220, 220, 255), -1)

    # Symmetrical circles
    cv2.circle(img, (center_x, center_y), 60, (255, 255, 255), -1)

    # Add slight noise for realism
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_them_frame(width=640, height=480):
    """
    Generate a synthetic "them" (outgroup) frame.
    Characteristics: dark, low-key lighting, asymmetrical, cool colors, shadowy.
    """
    # Start with dark background
    img = np.ones((height, width, 3), dtype=np.uint8) * 60

    # Add cool color cast (slight blue tint)
    img[:, :, 0] = np.clip(img[:, :, 0] + 30, 0, 255)  # Blue channel
    img[:, :, 1] = np.clip(img[:, :, 1] + 10, 0, 255)  # Green channel
    img[:, :, 2] = np.clip(img[:, :, 2] + 5, 0, 255)   # Red channel

    # Add asymmetrical, chaotic shapes
    # Random dark patches (shadows)
    for _ in range(5):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        size = np.random.randint(30, 80)
        cv2.circle(img, (x, y), size, (30, 35, 40), -1)

    # Asymmetrical rectangles
    cv2.rectangle(img, (100, 150), (250, 300), (80, 80, 90), -1)
    cv2.rectangle(img, (400, 100), (500, 350), (70, 75, 80), -1)

    # Add more noise for chaotic feel
    noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_test_dataset(output_dir, num_us=30, num_them=30):
    """
    Generate a complete test dataset with annotations.

    Args:
        output_dir: Directory to save frames and annotations
        num_us: Number of "us" frames to generate
        num_them: Number of "them" frames to generate

    Returns:
        Path to annotations CSV
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    annotations = []

    print(f"Generating {num_us} 'us' frames...")
    for i in range(num_us):
        img = generate_us_frame()
        frame_path = frames_dir / f"us_frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), img)
        annotations.append({
            'frame_path': str(frame_path.resolve()),
            'label': 'us',
            'confidence': 1.0,
            'notes': 'Synthetic test data',
            'annotator': 'auto'
        })

    print(f"Generating {num_them} 'them' frames...")
    for i in range(num_them):
        img = generate_them_frame()
        frame_path = frames_dir / f"them_frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), img)
        annotations.append({
            'frame_path': str(frame_path.resolve()),
            'label': 'them',
            'confidence': 1.0,
            'notes': 'Synthetic test data',
            'annotator': 'auto'
        })

    # Save annotations
    annotations_file = output_dir / "test_annotations.csv"
    df = pd.DataFrame(annotations)
    df.to_csv(annotations_file, index=False)

    print(f"\nGenerated {len(annotations)} frames")
    print(f"Frames saved to: {frames_dir}")
    print(f"Annotations saved to: {annotations_file}")

    return annotations_file


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test_data"
    num_us = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    num_them = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    generate_test_dataset(output_dir, num_us, num_them)
    print("\nTest data generation complete!")
    print("\nNext steps:")
    print(f"1. Extract features: python src/feature_extraction.py {output_dir}/frames {output_dir}/test_features.csv")
    print(f"2. Train model: python src/model.py {output_dir}/test_features.csv {output_dir}/test_annotations.csv results/test")
