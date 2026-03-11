"""
Filter extracted frames: keep only those containing people, delete the rest.

Usage:
    python filter_people_frames.py

Requires:
    pip install ultralytics
"""

import os
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# >>> CONFIGURATION — adjust these as needed <<<
# ============================================================

# Folder containing the extracted frame images
FRAMES_DIR = "./data/frames/heimkehr"

# Confidence threshold for person detections (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.5

# YOLO model size: "yolov8n" (nano/fastest), "yolov8s" (small),
#                  "yolov8m" (medium), "yolov8l" (large), "yolov8x" (largest/slowest)
MODEL_NAME = "yolov8n.pt"

# ============================================================


PERSON_CLASS_ID = 0  # COCO class 0 = person


def main():
    model = YOLO(MODEL_NAME)

    frames_path = Path(FRAMES_DIR)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted(
        f for f in frames_path.iterdir() if f.suffix.lower() in image_extensions
    )

    if not image_files:
        print(f"No images found in '{FRAMES_DIR}'.")
        return

    print(f"Found {len(image_files)} images. Running person detection...")

    kept = 0
    deleted = 0

    for img_path in image_files:
        results = model(str(img_path), verbose=False)[0]

        # Check if any detection is a person above the threshold
        has_person = False
        for box in results.boxes:
            if int(box.cls) == PERSON_CLASS_ID and float(box.conf) >= CONFIDENCE_THRESHOLD:
                has_person = True
                break

        if has_person:
            kept += 1
        else:
            os.remove(img_path)
            deleted += 1

    print(f"Done. Kept {kept} frames, deleted {deleted} frames.")


if __name__ == "__main__":
    main()