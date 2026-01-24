import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets


class FrameAnnotator:
    """Simple CSV-based frame annotation system."""

    def __init__(self, annotation_file):
        """
        Initialize annotator.

        Args:
            annotation_file: Path to CSV file for storing annotations
        """
        self.annotation_file = Path(annotation_file)
        self.annotation_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing annotations or create new
        if self.annotation_file.exists():
            self.annotations = pd.read_csv(self.annotation_file)
        else:
            self.annotations = pd.DataFrame(columns=[
                'frame_path', 'label', 'confidence', 'notes', 'annotator'
            ])

    def add_annotation(self, frame_path, label, confidence=1.0, notes="", annotator=""):
        """
        Add or update an annotation.

        Args:
            frame_path: Path to the frame image
            label: Label ('us', 'them', 'neutral', 'unclear')
            confidence: Confidence in annotation (0-1)
            notes: Optional notes
            annotator: Name of annotator
        """
        frame_path = str(Path(frame_path).resolve())

        # Remove existing annotation for this frame if exists
        self.annotations = self.annotations[self.annotations['frame_path'] != frame_path]

        # Add new annotation
        new_row = pd.DataFrame([{
            'frame_path': frame_path,
            'label': label,
            'confidence': confidence,
            'notes': notes,
            'annotator': annotator
        }])

        self.annotations = pd.concat([self.annotations, new_row], ignore_index=True)

    def get_annotation(self, frame_path):
        """Get annotation for a specific frame."""
        frame_path = str(Path(frame_path).resolve())
        result = self.annotations[self.annotations['frame_path'] == frame_path]
        if len(result) > 0:
            return result.iloc[0].to_dict()
        return None

    def save(self):
        """Save annotations to CSV."""
        self.annotations.to_csv(self.annotation_file, index=False)
        print(f"Saved {len(self.annotations)} annotations to {self.annotation_file}")

    def get_statistics(self):
        """Get annotation statistics."""
        stats = {
            'total': len(self.annotations),
            'by_label': self.annotations['label'].value_counts().to_dict(),
            'mean_confidence': self.annotations['confidence'].mean()
        }
        return stats

    def get_labeled_frames(self, label=None):
        """
        Get all frames with a specific label.

        Args:
            label: Filter by label (None = all)

        Returns:
            DataFrame of matching annotations
        """
        if label is None:
            return self.annotations
        return self.annotations[self.annotations['label'] == label]

    def export_for_training(self, output_file=None, binary_labels=True):
        """
        Export annotations in format suitable for ML training.

        Args:
            output_file: Optional CSV file to save to
            binary_labels: Convert to binary us=1, them=0 (removes neutral/unclear)

        Returns:
            DataFrame with frame_path and label columns
        """
        df = self.annotations[['frame_path', 'label']].copy()

        if binary_labels:
            # Filter to only us/them, convert to binary
            df = df[df['label'].isin(['us', 'them'])]
            df['label'] = df['label'].map({'us': 1, 'them': 0})

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Exported {len(df)} training samples to {output_file}")

        return df


def create_annotation_template(frame_dir, output_file):
    """
    Create annotation template CSV from a directory of frames.

    Args:
        frame_dir: Directory containing frame images
        output_file: Output CSV file path
    """
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))

    df = pd.DataFrame({
        'frame_path': [str(f.resolve()) for f in frames],
        'label': [''],
        'confidence': [1.0] * len(frames),
        'notes': [''],
        'annotator': ['']
    })

    df.to_csv(output_file, index=False)
    print(f"Created annotation template with {len(frames)} frames at {output_file}")
    return df


def show_frame(frame_path, figsize=(10, 6)):
    """Display a frame for annotation."""
    img = cv2.imread(str(frame_path))
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=figsize)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(Path(frame_path).name)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Could not load image: {frame_path}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "create-template":
            frame_dir = sys.argv[2]
            output = sys.argv[3] if len(sys.argv) > 3 else "data/annotated/annotations.csv"
            create_annotation_template(frame_dir, output)
        else:
            print("Usage: python annotation.py create-template <frame_dir> [output.csv]")
    else:
        print("Annotation module loaded. Use from Python/Jupyter notebook.")
