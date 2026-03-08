import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class VisualFeatureExtractor:
    

    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.feature_names = [
            # Lighting features
            'mean_brightness',
            'brightness_std',
            'contrast',
            'low_key_ratio',      # Proportion of dark pixels
            'high_key_ratio',     # Proportion of bright pixels

            # Color features
            'saturation_mean',
            'saturation_std',
            'hue_mean',
            'hue_std',
            'color_temperature',  # Warm vs cool tones

            # Composition features
            'edge_density',
            'center_brightness',   # Brightness in center vs edges
            'vertical_symmetry',
            'horizontal_symmetry',

            # Texture features
            'texture_contrast',
            'texture_homogeneity',

            # Face/figure detection (simplified)
            'dark_regions_count',
            'bright_regions_count',

            # Face detection features
            'face_count',
            'face_area_ratio',
            'largest_face_y_position',

            # Depth of field proxy
            'dof_variance',
        ]

    def extract_features(self, image_path):

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Crop bottom 12% to remove subtitles
        h, w = img.shape[:2]
        img = img[0:int(h*0.88), :]  # Keep top 88%, drop bottom 12%

        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        features = {}

        # Lighting features
        features['mean_brightness'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['contrast'] = np.std(gray) / (np.mean(gray) + 1e-7)

        # Low-key (dark) and high-key (bright) lighting
        features['low_key_ratio'] = np.sum(gray < 85) / gray.size
        features['high_key_ratio'] = np.sum(gray > 170) / gray.size

        # Color features
        h, s, v = cv2.split(hsv)
        features['saturation_mean'] = np.mean(s)
        features['saturation_std'] = np.std(s)
        features['hue_mean'] = np.mean(h)
        features['hue_std'] = np.std(h)

        # Color temperature (warm vs cool)
        b, g, r = cv2.split(img)
        features['color_temperature'] = np.mean(r) / (np.mean(b) + np.mean(g) + 1e-7)

        # Composition features
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / edges.size

        # Center vs edge brightness
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        features['center_brightness'] = np.mean(center_region) - np.mean(gray)

        # Symmetry
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        features['vertical_symmetry'] = 1.0 - np.mean(np.abs(
            left_half[:, :min_width].astype(float) - right_half[:, :min_width].astype(float)
        )) / 255.0

        top_half = gray[:h//2, :]
        bottom_half = cv2.flip(gray[h//2:, :], 0)
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        features['horizontal_symmetry'] = 1.0 - np.mean(np.abs(
            top_half[:min_height, :].astype(float) - bottom_half[:min_height, :].astype(float)
        )) / 255.0

        # Texture features using GLCM approximation
        features['texture_contrast'] = self._compute_local_contrast(gray)
        features['texture_homogeneity'] = self._compute_homogeneity(gray)

        # Region detection (dark vs bright regions)
        _, binary_dark = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        _, binary_bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        contours_dark, _ = cv2.findContours(binary_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_bright, _ = cv2.findContours(binary_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features['dark_regions_count'] = len([c for c in contours_dark if cv2.contourArea(c) > 100])
        features['bright_regions_count'] = len([c for c in contours_bright if cv2.contourArea(c) > 100])

        # Face detection features
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        features['face_count'] = len(faces)

        if len(faces) > 0:
            # Calculate total face area ratio
            total_face_area = sum([w * h for (x, y, w, h) in faces])
            features['face_area_ratio'] = total_face_area / (gray.shape[0] * gray.shape[1])

            # Position of largest face (vertical, normalized)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            features['largest_face_y_position'] = (y + h/2) / gray.shape[0]
        else:
            features['face_area_ratio'] = 0.0
            features['largest_face_y_position'] = 0.5  # Default to center

        # Depth of field proxy (sharpness variance across regions)
        h_img, w_img = gray.shape
        regions_sharpness = []
        for i in range(3):
            for j in range(3):
                region = gray[i*h_img//3:(i+1)*h_img//3, j*w_img//3:(j+1)*w_img//3]
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                regions_sharpness.append(np.var(laplacian))
        features['dof_variance'] = np.std(regions_sharpness)

        return features

    def _compute_local_contrast(self, gray):
        
        kernel_size = 15
        mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
        mean_sq = cv2.blur((gray.astype(float) ** 2), (kernel_size, kernel_size))
        std = np.sqrt(np.abs(mean_sq - mean ** 2))
        return np.mean(std)

    def _compute_homogeneity(self, gray):
        
        kernel_size = 5
        local_std = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
        variation = np.std(local_std)
        return 1.0 / (1.0 + variation)

    def extract_batch(self, image_paths, output_csv=None):
        
        results = []

        for img_path in tqdm(image_paths, desc="Extracting features"):
            try:
                features = self.extract_features(img_path)
                features['frame_path'] = str(Path(img_path).resolve())
                results.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        df = pd.DataFrame(results)

        # Reorder columns: frame_path first, then features
        cols = ['frame_path'] + self.feature_names
        df = df[cols]

        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Saved features to {output_csv}")

        return df

def extract_features_from_directory(input_dir, output_csv):
    
    input_dir = Path(input_dir)
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))

    print(f"Found {len(image_paths)} images in {input_dir}")

    extractor = VisualFeatureExtractor()
    df = extractor.extract_batch(image_paths, output_csv)

    print(f"Extracted {len(df)} feature vectors")
    return df

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        input_dir = sys.argv[1]
        output_csv = sys.argv[2]
        extract_features_from_directory(input_dir, output_csv)
    else:
        print("Usage: python feature_extraction.py <input_dir> <output.csv>")
