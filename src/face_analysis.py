"""
Face detection and analysis module for propaganda film analysis.
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: MTCNN not installed. Run: pip install mtcnn")


class FaceAnalyzer:
    """
    Detect faces and extract face-specific visual features.

    Designed for analyzing facial presentation differences in propaganda films:
    - "Us" hypothesis: Bright, heroic lighting from below, centered, individual close-ups
    - "Them" hypothesis: Dark/harsh lighting from above, off-center, crowds
    """

    def __init__(self, detector='mtcnn', min_confidence=0.90):
        """
        Initialize face detector.

        Args:
            detector: 'mtcnn' or 'opencv' (fallback)
            min_confidence: Minimum detection confidence (0.0-1.0)
        """
        self.min_confidence = min_confidence
        self.detector_type = detector

        if detector == 'mtcnn' and MTCNN_AVAILABLE:
            self.detector = MTCNN()
            print(f"Initialized MTCNN face detector (min_confidence={min_confidence})")
        else:
            # Fallback to OpenCV Haar Cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.detector_type = 'opencv'
            print("Initialized OpenCV Haar Cascade (fallback)")

    def detect_faces(self, image):
        """
        Detect all faces in an image.

        Args:
            image: OpenCV image (BGR format)

        Returns:
            List of face dictionaries with 'box' and 'confidence'
        """
        if self.detector_type == 'mtcnn':
            return self._detect_mtcnn(image)
        else:
            return self._detect_opencv(image)

    def _detect_mtcnn(self, image):
        """Detect faces using MTCNN."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)

        # Filter by confidence
        faces = []
        for det in detections:
            if det['confidence'] >= self.min_confidence:
                faces.append({
                    'box': det['box'],
                    'confidence': det['confidence'],
                    'keypoints': det.get('keypoints', {})
                })

        return faces

    def _detect_opencv(self, image):
        """Detect faces using OpenCV Haar Cascades (fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        faces = []
        for (x, y, w, h) in detections:
            faces.append({
                'box': [x, y, w, h],
                'confidence': 1.0,  # OpenCV doesn't provide confidence
                'keypoints': {}
            })

        return faces

    def extract_face_features(self, image, face_bbox):
        """
        Extract visual features from a face region.

        Features designed to capture propaganda cinematography:
        - Lighting direction (top-lit vs bottom-lit)
        - Face prominence (size, centrality)
        - Contrast with background
        - Basic composition metrics

        Args:
            image: Full frame image
            face_bbox: [x, y, w, h] bounding box

        Returns:
            Dictionary of face-specific features
        """
        x, y, w, h = face_bbox

        # Ensure bbox is within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return None

        # Extract face region
        face_crop = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = {}

        # === LIGHTING FEATURES ===
        features['face_brightness'] = np.mean(face_gray)
        features['face_contrast'] = np.std(face_gray)
        features['face_low_key_ratio'] = np.sum(face_gray < 85) / face_gray.size
        features['face_high_key_ratio'] = np.sum(face_gray > 170) / face_gray.size

        # Lighting direction (top-lit vs bottom-lit)
        # Positive = top-lit (harsh, sinister), Negative = bottom-lit (heroic)
        top_half = face_gray[:h//2, :]
        bottom_half = face_gray[h//2:, :]
        features['lighting_direction'] = (np.mean(top_half) - np.mean(bottom_half)) / 255.0

        # Left-right lighting asymmetry
        left_half = face_gray[:, :w//2]
        right_half = face_gray[:, w//2:]
        features['lighting_asymmetry'] = abs(np.mean(left_half) - np.mean(right_half)) / 255.0

        # === COMPOSITION FEATURES ===
        # Face size relative to frame
        face_area = w * h
        frame_area = img_w * img_h
        features['face_size_ratio'] = face_area / frame_area

        # Face centrality (1.0 = perfectly centered, 0.0 = at edge)
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        frame_center_x = img_w / 2
        frame_center_y = img_h / 2

        distance_from_center = np.sqrt(
            ((face_center_x - frame_center_x) / img_w) ** 2 +
            ((face_center_y - frame_center_y) / img_h) ** 2
        )
        max_distance = np.sqrt(0.5 ** 2 + 0.5 ** 2)  # Corner to center
        features['face_centrality'] = 1.0 - (distance_from_center / max_distance)

        # Face aspect ratio (facial framing)
        features['face_aspect_ratio'] = w / h if h > 0 else 1.0

        # === CONTRAST WITH BACKGROUND ===
        # Create mask for background (everything except face)
        mask = np.ones(frame_gray.shape, dtype=bool)
        mask[y:y+h, x:x+w] = False

        if np.sum(mask) > 0:
            background_brightness = np.mean(frame_gray[mask])
            features['face_bg_brightness_diff'] = (features['face_brightness'] - background_brightness) / 255.0
            features['face_bg_contrast'] = abs(features['face_brightness'] - background_brightness) / 255.0
        else:
            features['face_bg_brightness_diff'] = 0.0
            features['face_bg_contrast'] = 0.0

        # === FACE SHARPNESS/CLARITY ===
        # Laplacian variance (measure of focus/sharpness)
        laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
        features['face_sharpness'] = laplacian.var()

        # Edge density in face
        edges = cv2.Canny(face_gray, 100, 200)
        features['face_edge_density'] = np.sum(edges > 0) / edges.size

        return features

    def analyze_frame(self, image_path, label=None):
        """
        Analyze a single frame: detect faces and extract features.

        Args:
            image_path: Path to image file
            label: Optional label ('us' or 'them')

        Returns:
            Dictionary with frame-level and face-level data
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        faces = self.detect_faces(image)

        result = {
            'frame_path': str(Path(image_path).resolve()),
            'label': label,
            'num_faces': len(faces),
            'faces': []
        }

        for i, face_det in enumerate(faces):
            bbox = face_det['box']
            confidence = face_det['confidence']

            features = self.extract_face_features(image, bbox)
            if features is not None:
                face_data = {
                    'face_id': i,
                    'bbox': bbox,
                    'confidence': confidence,
                    **features
                }
                result['faces'].append(face_data)

        return result

    def analyze_dataset(self, image_paths, labels=None, output_csv=None):
        """
        Analyze multiple frames and extract face-level features.

        Args:
            image_paths: List of image paths
            labels: Optional list of labels corresponding to images
            output_csv: Optional path to save results as CSV

        Returns:
            DataFrame with one row per detected face
        """
        if labels is None:
            labels = [None] * len(image_paths)

        all_faces = []
        frame_stats = []

        print(f"Analyzing {len(image_paths)} frames...")
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                result = self.analyze_frame(img_path, label)

                # Frame-level stats
                frame_stats.append({
                    'frame_path': result['frame_path'],
                    'label': result['label'],
                    'num_faces': result['num_faces']
                })

                # Face-level data
                for face in result['faces']:
                    face_row = {
                        'frame_path': result['frame_path'],
                        'label': result['label'],
                        **face
                    }
                    all_faces.append(face_row)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        df_faces = pd.DataFrame(all_faces)
        df_frames = pd.DataFrame(frame_stats)

        print(f"\nDetected {len(df_faces)} faces across {len(df_frames)} frames")
        print(f"Frames with faces: {np.sum(df_frames['num_faces'] > 0)} / {len(df_frames)} ({100*np.sum(df_frames['num_faces'] > 0)/len(df_frames):.1f}%)")

        if output_csv and len(df_faces) > 0:
            df_faces.to_csv(output_csv, index=False)
            print(f"Saved face features to {output_csv}")

            # Save frame stats separately
            from pathlib import Path
            output_path = Path(output_csv)
            frame_csv = output_path.parent / (output_path.stem + '_frame_stats' + output_path.suffix)
            df_frames.to_csv(frame_csv, index=False)
            print(f"Saved frame statistics to {frame_csv}")

        return df_faces, df_frames

    def visualize_detections(self, image_path, output_path=None):
        """
        Visualize face detections with bounding boxes and feature annotations.

        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization

        Returns:
            Annotated image
        """
        image = cv2.imread(str(image_path))
        faces = self.detect_faces(image)

        vis_image = image.copy()

        for i, face_det in enumerate(faces):
            x, y, w, h = face_det['box']
            conf = face_det['confidence']

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract features for annotation
            features = self.extract_face_features(image, [x, y, w, h])
            if features:
                # Annotate with key features
                text = f"Face {i+1} (conf={conf:.2f})"
                cv2.putText(vis_image, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show lighting direction
                lighting = features['lighting_direction']
                lighting_text = "Top-lit" if lighting > 0 else "Bottom-lit"
                cv2.putText(vis_image, lighting_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if output_path:
            cv2.imwrite(str(output_path), vis_image)

        return vis_image


def compare_us_vs_them(df_faces, feature_names=None):
    """
    Statistical comparison of face features between 'us' and 'them' categories.

    Args:
        df_faces: DataFrame with face features and labels
        feature_names: Optional list of feature names to compare

    Returns:
        DataFrame with statistical test results
    """
    from scipy import stats

    if feature_names is None:
        # Default: all numerical features except metadata
        feature_names = [col for col in df_faces.columns
                        if col not in ['frame_path', 'label', 'face_id', 'bbox', 'confidence']]

    us_faces = df_faces[df_faces['label'] == 'us']
    them_faces = df_faces[df_faces['label'] == 'them']

    print(f"\n{'='*60}")
    print(f"FACE-LEVEL COMPARISON: 'Us' vs 'Them'")
    print(f"{'='*60}")
    print(f"'Us' faces: {len(us_faces)}")
    print(f"'Them' faces: {len(them_faces)}")
    print(f"{'='*60}\n")

    results = []

    for feature in feature_names:
        if feature not in df_faces.columns:
            continue

        us_values = us_faces[feature].dropna()
        them_values = them_faces[feature].dropna()

        if len(us_values) < 2 or len(them_values) < 2:
            continue

        # T-test
        t_stat, p_value = stats.ttest_ind(us_values, them_values)

        # Effect size (Cohen's d)
        mean_diff = us_values.mean() - them_values.mean()
        pooled_std = np.sqrt((us_values.std()**2 + them_values.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': feature,
            'us_mean': us_values.mean(),
            'us_std': us_values.std(),
            'them_mean': them_values.mean(),
            'them_std': them_values.std(),
            'mean_diff': mean_diff,
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('p_value')

    return df_results


if __name__ == "__main__":
    print("Face Analysis Module")
    print("=" * 60)
    print("Usage: from src.face_analysis import FaceAnalyzer")
    print("\nExample:")
    print("  analyzer = FaceAnalyzer()")
    print("  result = analyzer.analyze_frame('path/to/image.jpg')")
    print("  print(f'Detected {result['num_faces']} faces')")
