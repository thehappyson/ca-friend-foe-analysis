import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.scene_detection import (
    SceneDetector,
    extract_frames_from_scenes,
    filter_scenes_by_duration,
    get_scene_statistics
)
from src.feature_extraction import VisualFeatureExtractor
from src.model import predict_from_features


def analyze_video_by_scenes(
    video_path: str,
    model_path: str,
    output_dir: str,
    scene_method: str = 'content',
    scene_threshold: float = 27.0,
    min_scene_duration: float = 1.0,
    frames_per_scene: int = 5,
    frame_method: str = 'uniform'
):
    """
    Perform complete scene-based analysis of a video.

    Args:
        video_path: Path to input video
        model_path: Path to trained classification model
        output_dir: Directory for all outputs
        scene_method: Scene detection method ('content', 'adaptive', 'threshold')
        scene_threshold: Sensitivity threshold for scene detection
        min_scene_duration: Minimum scene length in seconds
        frames_per_scene: Number of frames to extract per scene
        frame_method: Frame extraction method ('uniform', 'keyframe')
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SCENE-BASED FRIEND-FOE ANALYSIS PIPELINE")
    print("="*70)
    print(f"Video: {video_path.name}")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Detect scenes
    print("STEP 1: Scene Detection")
    print("-" * 70)
    detector = SceneDetector(method=scene_method, threshold=scene_threshold)
    scenes = detector.detect_scenes(str(video_path), min_scene_len=min_scene_duration)

    # Save scene information
    scenes_csv = output_dir / f"{video_path.stem}_scenes.csv"
    detector.save_scenes_to_csv(scenes, scenes_csv)

    # Print statistics
    stats = get_scene_statistics(scenes)
    print("\nScene Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Step 2: Extract frames from scenes
    print("STEP 2: Frame Extraction")
    print("-" * 70)
    frames_dir = output_dir / "frames"
    scene_frames = extract_frames_from_scenes(
        str(video_path),
        scenes,
        str(frames_dir),
        frames_per_scene=frames_per_scene,
        method=frame_method
    )
    print()

    # Step 3: Extract features from frames
    print("STEP 3: Feature Extraction")
    print("-" * 70)
    extractor = VisualFeatureExtractor()

    all_features = []
    scene_features_map = {}

    for scene_id, frame_paths in tqdm(scene_frames.items(), desc="Extracting features"):
        scene_frame_features = []

        for frame_path in frame_paths:
            features = extractor.extract_features(frame_path)
            features['scene_id'] = scene_id
            features['frame_path'] = str(frame_path)
            all_features.append(features)
            scene_frame_features.append(features)

        scene_features_map[scene_id] = scene_frame_features

    # Save frame-level features
    features_df = pd.DataFrame(all_features)
    features_csv = output_dir / f"{video_path.stem}_frame_features.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"Saved frame features to {features_csv}")
    print()

    # Step 4: Classify each scene
    print("STEP 4: Scene Classification")
    print("-" * 70)

    scene_predictions = []

    for scene in tqdm(scenes, desc="Classifying scenes"):
        scene_id = scene.scene_id

        if scene_id not in scene_features_map:
            continue

        # Get features for all frames in this scene
        frame_features_list = scene_features_map[scene_id]

        if not frame_features_list:
            continue

        # Prepare features for prediction (exclude non-feature columns)
        feature_cols = extractor.feature_names
        frame_features_array = []

        for feat_dict in frame_features_list:
            frame_feat = [feat_dict[col] for col in feature_cols]
            frame_features_array.append(frame_feat)

        frame_features_array = np.array(frame_features_array)

        # Predict for each frame in the scene
        predictions = predict_from_features(frame_features_array, model_path)

        # Aggregate predictions for the scene
        # Use majority vote or average probability
        foe_votes = np.sum(predictions == 1)
        friend_votes = np.sum(predictions == 0)
        foe_ratio = foe_votes / len(predictions)

        # Determine scene classification (majority vote)
        scene_prediction = 1 if foe_votes > friend_votes else 0
        scene_label = "Foe" if scene_prediction == 1 else "Friend"

        scene_result = {
            'scene_id': scene_id,
            'start_time': scene.start_time,
            'end_time': scene.end_time,
            'duration': scene.duration,
            'frame_count': scene.frame_count,
            'analyzed_frames': len(predictions),
            'foe_votes': int(foe_votes),
            'friend_votes': int(friend_votes),
            'foe_ratio': foe_ratio,
            'prediction': scene_prediction,
            'label': scene_label,
            'confidence': max(foe_ratio, 1 - foe_ratio)
        }

        scene_predictions.append(scene_result)

    # Save scene-level predictions
    scene_results_df = pd.DataFrame(scene_predictions)
    scene_results_csv = output_dir / f"{video_path.stem}_scene_predictions.csv"
    scene_results_df.to_csv(scene_results_csv, index=False)
    print(f"Saved scene predictions to {scene_results_csv}")
    print()

    # Step 5: Generate summary report
    print("STEP 5: Summary Report")
    print("="*70)

    total_scenes = len(scene_predictions)
    foe_scenes = sum(1 for s in scene_predictions if s['prediction'] == 1)
    friend_scenes = total_scenes - foe_scenes

    total_duration = sum(s['duration'] for s in scene_predictions)
    foe_duration = sum(s['duration'] for s in scene_predictions if s['prediction'] == 1)
    friend_duration = total_duration - foe_duration

    print(f"Total Scenes Analyzed: {total_scenes}")
    print(f"  Friend Scenes: {friend_scenes} ({friend_scenes/total_scenes*100:.1f}%)")
    print(f"  Foe Scenes: {foe_scenes} ({foe_scenes/total_scenes*100:.1f}%)")
    print()
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"  Friend Duration: {friend_duration:.2f}s ({friend_duration/total_duration*100:.1f}%)")
    print(f"  Foe Duration: {foe_duration:.2f}s ({foe_duration/total_duration*100:.1f}%)")
    print()

    # Most confident predictions
    print("Most Confident Foe Scenes:")
    foe_scenes_sorted = sorted(
        [s for s in scene_predictions if s['prediction'] == 1],
        key=lambda x: x['confidence'],
        reverse=True
    )[:5]
    for scene in foe_scenes_sorted:
        print(f"  Scene {scene['scene_id']}: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s "
              f"(confidence: {scene['confidence']:.2%})")
    print()

    print("Most Confident Friend Scenes:")
    friend_scenes_sorted = sorted(
        [s for s in scene_predictions if s['prediction'] == 0],
        key=lambda x: x['confidence'],
        reverse=True
    )[:5]
    for scene in friend_scenes_sorted:
        print(f"  Scene {scene['scene_id']}: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s "
              f"(confidence: {scene['confidence']:.2%})")
    print()

    # Save summary report
    report_path = output_dir / f"{video_path.stem}_report.txt"
    with open(report_path, 'w') as f:
        f.write("SCENE-BASED FRIEND-FOE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Video: {video_path.name}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")

        f.write("SETTINGS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Scene Detection Method: {scene_method}\n")
        f.write(f"Scene Threshold: {scene_threshold}\n")
        f.write(f"Minimum Scene Duration: {min_scene_duration}s\n")
        f.write(f"Frames per Scene: {frames_per_scene}\n\n")

        f.write("RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Scenes: {total_scenes}\n")
        f.write(f"Friend Scenes: {friend_scenes} ({friend_scenes/total_scenes*100:.1f}%)\n")
        f.write(f"Foe Scenes: {foe_scenes} ({foe_scenes/total_scenes*100:.1f}%)\n\n")
        f.write(f"Total Duration: {total_duration:.2f}s\n")
        f.write(f"Friend Duration: {friend_duration:.2f}s ({friend_duration/total_duration*100:.1f}%)\n")
        f.write(f"Foe Duration: {foe_duration:.2f}s ({foe_duration/total_duration*100:.1f}%)\n\n")

        f.write("SCENE DETAILS\n")
        f.write("-" * 70 + "\n")
        for scene in scene_predictions:
            f.write(f"Scene {scene['scene_id']:3d}: {scene['start_time']:7.2f}s - {scene['end_time']:7.2f}s | "
                   f"{scene['label']:6s} (conf: {scene['confidence']:.2%})\n")

    print(f"Report saved to {report_path}")
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return scene_results_df


def main():
    parser = argparse.ArgumentParser(
        description='Scene-based friend-foe analysis pipeline'
    )
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained classification model')
    parser.add_argument('--output', type=str, default='results/scene_analysis',
                       help='Output directory')
    parser.add_argument('--scene-method', type=str, default='content',
                       choices=['content', 'adaptive', 'threshold'],
                       help='Scene detection method')
    parser.add_argument('--scene-threshold', type=float, default=27.0,
                       help='Scene detection threshold')
    parser.add_argument('--min-scene-duration', type=float, default=1.0,
                       help='Minimum scene duration in seconds')
    parser.add_argument('--frames-per-scene', type=int, default=5,
                       help='Number of frames to extract per scene')
    parser.add_argument('--frame-method', type=str, default='uniform',
                       choices=['uniform', 'keyframe'],
                       help='Frame extraction method')

    args = parser.parse_args()

    analyze_video_by_scenes(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        scene_method=args.scene_method,
        scene_threshold=args.scene_threshold,
        min_scene_duration=args.min_scene_duration,
        frames_per_scene=args.frames_per_scene,
        frame_method=args.frame_method
    )


if __name__ == '__main__':
    main()
