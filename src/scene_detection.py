import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("Warning: scenedetect not available. Install with: pip install scenedetect[opencv]")


@dataclass
class Scene:
    """Represents a detected scene in a video."""
    scene_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    frame_count: int

    def __repr__(self):
        return (f"Scene {self.scene_id}: frames {self.start_frame}-{self.end_frame} "
                f"({self.start_time:.2f}s - {self.end_time:.2f}s, {self.duration:.2f}s)")


class SceneDetector:
    """
    Detect scenes and shots in videos using various algorithms.

    Methods:
        - Content-based detection (detects cuts based on content changes)
        - Adaptive detection (more sophisticated, adapts to video characteristics)
        - Threshold-based detection (simple frame difference method)
    """

    def __init__(self, method='content', threshold=27.0):
        """
        Initialize scene detector.

        Args:
            method: Detection method ('content', 'adaptive', or 'threshold')
            threshold: Sensitivity threshold for detection (lower = more sensitive)
        """
        self.method = method
        self.threshold = threshold

    def detect_scenes(self, video_path: str, min_scene_len: float = 1.0) -> List[Scene]:
        """
        Detect scenes in a video.

        Args:
            video_path: Path to video file
            min_scene_len: Minimum scene length in seconds

        Returns:
            List of Scene objects
        """
        video_path = Path(video_path)

        if self.method in ['content', 'adaptive'] and SCENEDETECT_AVAILABLE:
            return self._detect_with_scenedetect(video_path, min_scene_len)
        else:
            if not SCENEDETECT_AVAILABLE:
                print(f"Falling back to threshold method (scenedetect not available)")
            return self._detect_with_threshold(video_path, min_scene_len)

    def _detect_with_scenedetect(self, video_path: Path, min_scene_len: float) -> List[Scene]:
        """Detect scenes using PySceneDetect library."""
        from scenedetect import detect, ContentDetector, AdaptiveDetector

        print(f"Detecting scenes in {video_path.name} using {self.method} method...")

        # Choose detector
        if self.method == 'adaptive':
            detector = AdaptiveDetector(
                adaptive_threshold=self.threshold,
                min_scene_len=min_scene_len
            )
        else:  # content
            detector = ContentDetector(
                threshold=self.threshold,
                min_scene_len=min_scene_len
            )

        # Detect scenes
        scene_list = detect(str(video_path), detector)

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Convert to Scene objects
        scenes = []
        for i, (start_time, end_time) in enumerate(scene_list):
            start_frame = int(start_time.get_frames())
            end_frame = int(end_time.get_frames())
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()

            scene = Scene(
                scene_id=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_sec,
                end_time=end_sec,
                duration=end_sec - start_sec,
                frame_count=end_frame - start_frame
            )
            scenes.append(scene)

        print(f"Detected {len(scenes)} scenes")
        return scenes

    def _detect_with_threshold(self, video_path: Path, min_scene_len: float) -> List[Scene]:
        """
        Detect scenes using simple frame difference threshold method.
        Fallback when scenedetect is not available.
        """
        print(f"Detecting scenes in {video_path.name} using threshold method...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_scene_frames = int(min_scene_len * fps)

        print(f"FPS: {fps:.2f}, Total frames: {total_frames}")

        # Detect cuts
        cut_frames = [0]  # Start with first frame
        prev_frame = None
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="Analyzing frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and resize for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))

            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                diff_score = np.mean(diff)

                # Detect cut if difference exceeds threshold
                if diff_score > self.threshold:
                    # Check minimum scene length
                    if len(cut_frames) == 0 or (frame_idx - cut_frames[-1]) >= min_scene_frames:
                        cut_frames.append(frame_idx)

            prev_frame = gray.copy()
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Add final frame
        cut_frames.append(total_frames - 1)

        # Create Scene objects
        scenes = []
        for i in range(len(cut_frames) - 1):
            start_frame = cut_frames[i]
            end_frame = cut_frames[i + 1]
            start_time = start_frame / fps
            end_time = end_frame / fps

            scene = Scene(
                scene_id=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                frame_count=end_frame - start_frame
            )
            scenes.append(scene)

        print(f"Detected {len(scenes)} scenes")
        return scenes

    def save_scenes_to_csv(self, scenes: List[Scene], output_path: str):
        """
        Save detected scenes to CSV file.

        Args:
            scenes: List of Scene objects
            output_path: Path to output CSV file
        """
        data = []
        for scene in scenes:
            data.append({
                'scene_id': scene.scene_id,
                'start_frame': scene.start_frame,
                'end_frame': scene.end_frame,
                'start_time': scene.start_time,
                'end_time': scene.end_time,
                'duration': scene.duration,
                'frame_count': scene.frame_count
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved scene information to {output_path}")
        return df


def extract_frames_from_scenes(
    video_path: str,
    scenes: List[Scene],
    output_dir: str,
    frames_per_scene: int = 5,
    method: str = 'uniform'
) -> Dict[int, List[Path]]:
    """
    Extract representative frames from each detected scene.

    Args:
        video_path: Path to video file
        scenes: List of Scene objects
        output_dir: Directory to save extracted frames
        frames_per_scene: Number of frames to extract per scene
        method: Extraction method ('uniform', 'keyframe', or 'all')

    Returns:
        Dictionary mapping scene_id to list of extracted frame paths
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    scene_frames = {}

    print(f"Extracting {frames_per_scene} frames per scene using '{method}' method...")

    for scene in tqdm(scenes, desc="Processing scenes"):
        frame_paths = []

        # Determine which frames to extract
        if method == 'uniform':
            # Uniformly distributed frames
            frame_indices = np.linspace(
                scene.start_frame,
                scene.end_frame - 1,
                min(frames_per_scene, scene.frame_count),
                dtype=int
            )
        elif method == 'keyframe':
            # Extract first, middle, and last frames
            if scene.frame_count <= frames_per_scene:
                frame_indices = range(scene.start_frame, scene.end_frame)
            else:
                step = scene.frame_count // frames_per_scene
                frame_indices = [scene.start_frame + i * step for i in range(frames_per_scene)]
        else:  # all
            frame_indices = range(scene.start_frame, scene.end_frame)

        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                timestamp = frame_idx / fps
                output_path = output_dir / (
                    f"{video_path.stem}_scene{scene.scene_id:03d}_"
                    f"frame{frame_idx:06d}_t{timestamp:.2f}s.jpg"
                )
                cv2.imwrite(str(output_path), frame)
                frame_paths.append(output_path)

        scene_frames[scene.scene_id] = frame_paths

    cap.release()

    total_frames = sum(len(paths) for paths in scene_frames.values())
    print(f"Extracted {total_frames} frames from {len(scenes)} scenes")

    return scene_frames


def filter_scenes_by_duration(
    scenes: List[Scene],
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None
) -> List[Scene]:
    """
    Filter scenes by duration criteria.

    Args:
        scenes: List of Scene objects
        min_duration: Minimum scene duration in seconds (None = no minimum)
        max_duration: Maximum scene duration in seconds (None = no maximum)

    Returns:
        Filtered list of Scene objects
    """
    filtered = scenes

    if min_duration is not None:
        filtered = [s for s in filtered if s.duration >= min_duration]

    if max_duration is not None:
        filtered = [s for s in filtered if s.duration <= max_duration]

    print(f"Filtered {len(scenes)} scenes to {len(filtered)} scenes")
    return filtered


def get_scene_statistics(scenes: List[Scene]) -> Dict:
    """
    Calculate statistics about detected scenes.

    Args:
        scenes: List of Scene objects

    Returns:
        Dictionary with scene statistics
    """
    if not scenes:
        return {}

    durations = [s.duration for s in scenes]
    frame_counts = [s.frame_count for s in scenes]

    stats = {
        'total_scenes': len(scenes),
        'mean_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'std_duration': np.std(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        'mean_frame_count': np.mean(frame_counts),
        'total_frames': sum(frame_counts)
    }

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Detect scenes in a video')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default='scenes.csv',
                       help='Output CSV file for scene information')
    parser.add_argument('--method', type=str, default='content',
                       choices=['content', 'adaptive', 'threshold'],
                       help='Detection method')
    parser.add_argument('--threshold', type=float, default=27.0,
                       help='Detection threshold')
    parser.add_argument('--min-scene-len', type=float, default=1.0,
                       help='Minimum scene length in seconds')
    parser.add_argument('--extract-frames', action='store_true',
                       help='Extract frames from detected scenes')
    parser.add_argument('--frames-per-scene', type=int, default=5,
                       help='Number of frames to extract per scene')
    parser.add_argument('--frame-output', type=str, default='frames',
                       help='Output directory for extracted frames')

    args = parser.parse_args()

    # Detect scenes
    detector = SceneDetector(method=args.method, threshold=args.threshold)
    scenes = detector.detect_scenes(args.video, min_scene_len=args.min_scene_len)

    # Print scene information
    print("\n" + "="*60)
    print("DETECTED SCENES")
    print("="*60)
    for scene in scenes:
        print(scene)

    # Calculate and print statistics
    stats = get_scene_statistics(scenes)
    print("\n" + "="*60)
    print("SCENE STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Save to CSV
    detector.save_scenes_to_csv(scenes, args.output)

    # Extract frames if requested
    if args.extract_frames:
        print("\n" + "="*60)
        print("EXTRACTING FRAMES")
        print("="*60)
        scene_frames = extract_frames_from_scenes(
            args.video,
            scenes,
            args.frame_output,
            frames_per_scene=args.frames_per_scene
        )
