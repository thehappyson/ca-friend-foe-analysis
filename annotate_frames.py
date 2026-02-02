import cv2
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def annotate_frames(frames_dir, output_csv, annotator_name="annotator", start_from=0):
    """
    Interactive frame annotation tool.

    Args:
        frames_dir: Directory containing frames to annotate
        output_csv: Path to save annotations CSV
        annotator_name: Your name/ID
        start_from: Frame index to start from (resume annotation)
    """
    frames_dir = Path(frames_dir)
    output_csv = Path(output_csv)

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    frame_paths = []
    for ext in image_extensions:
        frame_paths.extend(frames_dir.glob(f"**/{ext}"))

    frame_paths = sorted(frame_paths)

    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    print("="*70)
    print("INTERACTIVE FRAME ANNOTATION TOOL")
    print("="*70)
    print(f"Found {len(frame_paths)} frames")
    print(f"Starting from frame {start_from}")
    print()
    print("CONTROLS:")
    print("  [U] or [F] → Label as FRIEND/US (positive, community, bright)")
    print("  [T] or [E] → Label as FOE/THEM (negative, threat, dark)")
    print("  [N] or [X] → Label as NEUTRAL (ambiguous, technical)")
    print("  [S] → SKIP this frame")
    print("  [B] → GO BACK to previous frame")
    print("  [Q] → QUIT and save")
    print()
    print("Press any key to start...")
    print("="*70)

    cv2.namedWindow('Frame Annotation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame Annotation', 1280, 720)
    cv2.waitKey(0)

    # Load existing annotations if resuming
    annotations = []
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        annotations = existing_df.to_dict('records')
        print(f"\nLoaded {len(annotations)} existing annotations")

    # Track annotated paths
    annotated_paths = {str(ann['frame_path']) for ann in annotations}

    idx = start_from
    confidence_map = {'u': 1.0, 'f': 1.0, 't': 1.0, 'e': 1.0, 'n': 0.5, 'x': 0.5}

    while idx < len(frame_paths):
        frame_path = frame_paths[idx]

        # Skip if already annotated
        if str(frame_path) in annotated_paths:
            idx += 1
            continue

        # Load and display frame
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Could not load {frame_path}")
            idx += 1
            continue

        # Add text overlay
        h, w = img.shape[:2]
        overlay = img.copy()

        # Semi-transparent black bar at top
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        # Progress text
        progress_text = f"Frame {idx+1}/{len(frame_paths)} | Already annotated: {len(annotations)}"
        cv2.putText(img, progress_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Filename
        cv2.putText(img, frame_path.name, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Frame Annotation', img)

        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        key_char = chr(key).lower() if key < 128 else ''

        if key_char in ['q', '\x1b']:  # Q or ESC
            print("\nQuitting...")
            break

        elif key_char in ['u', 'f']:  # Friend/Us
            annotations.append({
                'frame_path': str(frame_path),
                'label': 'us',
                'confidence': confidence_map[key_char],
                'notes': 'Friend/Us - manually annotated',
                'annotator': annotator_name,
                'timestamp': datetime.now().isoformat()
            })
            annotated_paths.add(str(frame_path))
            print(f"[{idx+1}] {frame_path.name} → FRIEND/US")
            idx += 1

        elif key_char in ['t', 'e']:  # Foe/Them
            annotations.append({
                'frame_path': str(frame_path),
                'label': 'them',
                'confidence': confidence_map[key_char],
                'notes': 'Foe/Them - manually annotated',
                'annotator': annotator_name,
                'timestamp': datetime.now().isoformat()
            })
            annotated_paths.add(str(frame_path))
            print(f"[{idx+1}] {frame_path.name} → FOE/THEM")
            idx += 1

        elif key_char in ['n', 'x']:  # Neutral
            annotations.append({
                'frame_path': str(frame_path),
                'label': 'neutral',
                'confidence': confidence_map[key_char],
                'notes': 'Neutral/Ambiguous - manually annotated',
                'annotator': annotator_name,
                'timestamp': datetime.now().isoformat()
            })
            annotated_paths.add(str(frame_path))
            print(f"[{idx+1}] {frame_path.name} → NEUTRAL")
            idx += 1

        elif key_char == 's':  # Skip
            print(f"[{idx+1}] {frame_path.name} → SKIPPED")
            idx += 1

        elif key_char == 'b':  # Back
            if idx > 0:
                # Remove last annotation if going back
                if annotations and str(frame_paths[idx-1]) == annotations[-1]['frame_path']:
                    removed = annotations.pop()
                    annotated_paths.discard(removed['frame_path'])
                    print(f"Removed annotation for {Path(removed['frame_path']).name}")
                idx -= 1
                print(f"Going back to frame {idx+1}")
            else:
                print("Already at first frame")

        else:
            print(f"Unknown key: {key_char} (press U/F for Friend, T/E for Foe, N for Neutral)")

    cv2.destroyAllWindows()

    # Save annotations
    if annotations:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(annotations)
        df.to_csv(output_csv, index=False)

        print("\n" + "="*70)
        print("ANNOTATION SUMMARY")
        print("="*70)
        print(f"Total annotated: {len(annotations)}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        print(f"\nSaved to: {output_csv}")
        print("="*70)
    else:
        print("\nNo annotations saved.")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive frame annotation tool'
    )
    parser.add_argument('--frames-dir', type=str, required=True,
                       help='Directory containing frames to annotate')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file for annotations')
    parser.add_argument('--annotator', type=str, default='annotator',
                       help='Your name/ID')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Frame index to start from (for resuming)')

    args = parser.parse_args()

    annotate_frames(
        args.frames_dir,
        args.output,
        args.annotator,
        args.start_from
    )


if __name__ == '__main__':
    main()
