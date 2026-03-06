#!/usr/bin/env python3
"""
Interactive frame viewer for annotation.
Shows frames and allows keyboard-based labeling.
"""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import sys

class FrameViewer:
    def __init__(self, frames_dir, annotation_csv):
        """Initialize frame viewer."""
        self.frames_dir = Path(frames_dir)
        self.annotation_csv = Path(annotation_csv)

        # Load or create annotation CSV
        if self.annotation_csv.exists():
            self.annotations = pd.read_csv(self.annotation_csv, dtype={'label': str, 'notes': str, 'annotator': str})
            # Replace NaN with empty string
            self.annotations['label'] = self.annotations['label'].fillna('')
            self.annotations['notes'] = self.annotations['notes'].fillna('')
            self.annotations['annotator'] = self.annotations['annotator'].fillna('')
            print(f"Loaded existing annotations: {len(self.annotations)} frames")
        else:
            # Create new annotation file
            frames = sorted(self.frames_dir.glob("*.png")) + sorted(self.frames_dir.glob("*.jpg"))
            self.annotations = pd.DataFrame({
                'frame_path': [str(f.resolve()) for f in frames],
                'label': [''] * len(frames),
                'confidence': [1.0] * len(frames),
                'notes': [''] * len(frames),
                'annotator': [''] * len(frames)
            })
            print(f"Created new annotation file with {len(self.annotations)} frames")

        self.current_idx = 0
        self.window_name = "Frame Annotator"

        # Find first unannotated frame
        for idx, row in self.annotations.iterrows():
            if row['label'] == '' or pd.isna(row['label']):
                self.current_idx = idx
                break

        print(f"\nStarting at frame {self.current_idx + 1}/{len(self.annotations)}")
        self.show_instructions()

    def show_instructions(self):
        """Display keyboard instructions."""
        print("\n" + "="*60)
        print("KEYBOARD SHORTCUTS:")
        print("="*60)
        print("  1 or U : Label as 'us' (ingroup)")
        print("  2 or T : Label as 'them' (outgroup)")
        print("  3 or B : Label as 'both' (both groups)")
        print("  4 or N : Label as 'neutral'")
        print("  5 or ? : Label as 'unclear'")
        print("")
        print("  → or D : Next frame (without labeling)")
        print("  ← or A : Previous frame")
        print("  S      : Save annotations")
        print("  Q      : Quit and save")
        print("  H      : Show this help")
        print("="*60 + "\n")

    def get_progress_stats(self):
        """Get annotation progress statistics."""
        total = len(self.annotations)
        labeled = len(self.annotations[self.annotations['label'].notna() & (self.annotations['label'] != '')])
        stats = self.annotations['label'].value_counts().to_dict()
        return total, labeled, stats

    def show_frame(self):
        """Display current frame with info overlay."""
        if self.current_idx >= len(self.annotations):
            print("\n✓ All frames reviewed!")
            return False

        row = self.annotations.iloc[self.current_idx]
        frame_path = row['frame_path']

        # Load frame
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not load {frame_path}")
            return True

        # Create display image with info overlay
        display_img = img.copy()
        h, w = display_img.shape[:2]

        # Add black bar at top for info
        bar_height = 80
        display_img = cv2.copyMakeBorder(display_img, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Get stats
        total, labeled, stats = self.get_progress_stats()
        progress_pct = (labeled / total) * 100

        # Add text info
        frame_name = Path(frame_path).name
        current_label = row['label'] if row['label'] and not pd.isna(row['label']) else 'unlabeled'

        # Info text
        info_lines = [
            f"Frame: {self.current_idx + 1}/{total}  |  Progress: {labeled}/{total} ({progress_pct:.1f}%)",
            f"File: {frame_name}",
            f"Current label: {current_label}",
        ]

        y_offset = 20
        for line in info_lines:
            cv2.putText(display_img, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25

        # Resize if too large
        max_height = 900
        if display_img.shape[0] > max_height:
            scale = max_height / display_img.shape[0]
            new_w = int(display_img.shape[1] * scale)
            display_img = cv2.resize(display_img, (new_w, max_height))

        cv2.imshow(self.window_name, display_img)
        return True

    def save_annotations(self):
        """Save annotations to CSV."""
        self.annotation_csv.parent.mkdir(parents=True, exist_ok=True)
        self.annotations.to_csv(self.annotation_csv, index=False)
        total, labeled, stats = self.get_progress_stats()
        print(f"\n✓ Saved! Progress: {labeled}/{total} frames labeled")
        print(f"  Labels: {stats}")

    def label_current(self, label):
        """Label current frame and move to next."""
        self.annotations.at[self.current_idx, 'label'] = label
        print(f"Frame {self.current_idx + 1} labeled as '{label}'")
        self.current_idx += 1
        return True

    def next_frame(self):
        """Move to next frame."""
        self.current_idx = min(self.current_idx + 1, len(self.annotations) - 1)

    def prev_frame(self):
        """Move to previous frame."""
        self.current_idx = max(self.current_idx - 1, 0)

    def run(self):
        """Run the interactive viewer."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        running = True
        while running:
            if not self.show_frame():
                break

            key = cv2.waitKey(0) & 0xFF

            # Labels
            if key == ord('1') or key == ord('u') or key == ord('U'):
                self.label_current('us')
            elif key == ord('2') or key == ord('t') or key == ord('T'):
                self.label_current('them')
            elif key == ord('3') or key == ord('b') or key == ord('B'):
                self.label_current('both')
            elif key == ord('4') or key == ord('n') or key == ord('N'):
                self.label_current('neutral')
            elif key == ord('5') or key == ord('?'):
                self.label_current('unclear')

            # Navigation
            elif key == 83 or key == ord('d') or key == ord('D'):  # Right arrow or D
                self.next_frame()
            elif key == 81 or key == ord('a') or key == ord('A'):  # Left arrow or A
                self.prev_frame()

            # Commands
            elif key == ord('s') or key == ord('S'):
                self.save_annotations()
            elif key == ord('q') or key == ord('Q'):
                self.save_annotations()
                print("\nQuitting...")
                running = False
            elif key == ord('h') or key == ord('H'):
                self.show_instructions()
            elif key == 27:  # ESC
                print("\nQuitting without saving...")
                running = False

        cv2.destroyAllWindows()

        # Final stats
        total, labeled, stats = self.get_progress_stats()
        print(f"\n{'='*60}")
        print(f"FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Total frames: {total}")
        print(f"Labeled: {labeled} ({(labeled/total)*100:.1f}%)")
        print(f"Unlabeled: {total - labeled}")
        print(f"\nLabel distribution:")
        for label, count in sorted(stats.items()):
            print(f"  {label}: {count}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python annotate_viewer.py <frames_dir> <annotation_csv>")
        print("\nExample:")
        print("  python annotate_viewer.py frames/jud_suess data/annotated/jud_suess_annotations.csv")
        sys.exit(1)

    frames_dir = sys.argv[1]
    annotation_csv = sys.argv[2]

    viewer = FrameViewer(frames_dir, annotation_csv)
    viewer.run()
