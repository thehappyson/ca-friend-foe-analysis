"""
Extract individual frames from an MP4 file based on frame numbers listed in a txt file.

Usage:
    python extract_frames.py <video_file> <frame_list.txt>

The txt file should have two frame numbers per line (space or tab separated).
Each number is treated as an individual frame to extract.
"""

import cv2
import sys
import os


def load_frame_numbers(txt_path: str) -> list[int]:
    """Read the txt file and collect all unique frame numbers."""
    frames = set()
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            for part in parts:
                frames.add(int(part))
    return sorted(frames)


def extract_frames(video_path: str, frame_numbers: list[int], output_dir: str):
    """Extract and save the specified frames as images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames total.")
    print(f"Extracting {len(frame_numbers)} frames...")

    os.makedirs(output_dir, exist_ok=True)

    frame_set = set(frame_numbers)
    max_frame = max(frame_numbers)
    current_frame = 0
    extracted = 0

    while cap.isOpened() and current_frame <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_set:
            out_path = os.path.join(output_dir, f"frame_{current_frame:06d}.png")
            cv2.imwrite(out_path, frame)
            extracted += 1

        current_frame += 1

    cap.release()
    print(f"Done. Extracted {extracted}/{len(frame_numbers)} frames to '{output_dir}'.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_frames.py <video.mp4> <frames.txt>")
        syspip.exit(1)

    video_path = sys.argv[1]
    txt_path = sys.argv[2]

    # ============================================================
    output_dir = "./data/frames"
    # ============================================================

    frame_numbers = load_frame_numbers(txt_path)
    extract_frames(video_path, frame_numbers, output_dir)