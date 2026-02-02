import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_frames_by_interval(video_path, output_dir, interval_seconds=1.0, max_frames=None):
    """
    Extract frames from video at regular time intervals.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        interval_seconds: Time interval between frames (default: 1 second)
        max_frames: Maximum number of frames to extract (None = all)

    Returns:
        List of extracted frame paths
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    frame_interval = int(fps * interval_seconds)

    print(f"Video: {video_path.name}")
    print(f"Duration: {duration:.2f}s, FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"Extracting every {interval_seconds}s (every {frame_interval} frames)")

    extracted_paths = []
    frame_count = 0
    extracted_count = 0

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            output_path = output_dir / f"{video_path.stem}_frame_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted_paths.append(output_path)
            extracted_count += 1

            if max_frames and extracted_count >= max_frames:
                break

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"Extracted {extracted_count} frames to {output_dir}")
    return extracted_paths


def extract_frames_uniform(video_path, output_dir, num_frames=100):
    """
    Extract a fixed number of uniformly distributed frames from video.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract

    Returns:
        List of extracted frame paths
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame indices to extract (uniformly distributed)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    print(f"Extracting {num_frames} uniformly distributed frames from {total_frames} total frames")

    extracted_paths = []

    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Extracting frames")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            timestamp = frame_idx / fps
            output_path = output_dir / f"{video_path.stem}_frame_{i:06d}_t{timestamp:.2f}s.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted_paths.append(output_path)

    cap.release()
    print(f"Extracted {len(extracted_paths)} frames to {output_dir}")
    return extracted_paths

## Wie funktioniert der capture und was sind die wichtigsten Eigenschaften
def get_video_info(video_path):
    """Get basic information about a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0

    cap.release()
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video files")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds")
    parser.add_argument("--uniform", type=int, help="Extract N uniformly distributed frames")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to extract")

    args = parser.parse_args()

    if args.uniform:
        extract_frames_uniform(args.video, args.output, args.uniform)
    else:
        extract_frames_by_interval(args.video, args.output, args.interval, args.max_frames)
