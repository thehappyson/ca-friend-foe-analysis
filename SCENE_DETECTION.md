# Scene Detection und Shot Analysis

## Übersicht

Das Scene Detection Modul ermöglicht die automatische Einteilung von Videos in Szenen und Shots. Dies schränkt den Scope der Klassifizierung ein und ermöglicht eine präzisere Analyse auf Szenen-Ebene.

## Features

### 1. Scene Detection Methoden

#### Content-Based Detection (empfohlen)
- Erkennt Schnitte basierend auf Inhaltsänderungen
- Nutzt PySceneDetect Library
- Robust gegen Kamerabewegungen und Beleuchtungsänderungen

```python
from src.scene_detection import SceneDetector

detector = SceneDetector(method='content', threshold=27.0)
scenes = detector.detect_scenes('video.mp4', min_scene_len=1.0)
```

#### Adaptive Detection
- Passt sich automatisch an Video-Charakteristika an
- Gut für Videos mit variabler Qualität
- Etwas langsamer als Content-Based

```python
detector = SceneDetector(method='adaptive', threshold=27.0)
scenes = detector.detect_scenes('video.mp4')
```

#### Threshold-Based Detection (Fallback)
- Einfache Frame-Differenz-Methode
- Funktioniert ohne PySceneDetect
- Schneller, aber weniger genau

```python
detector = SceneDetector(method='threshold', threshold=30.0)
scenes = detector.detect_scenes('video.mp4')
```

### 2. Frame Extraction aus Szenen

```python
from src.scene_detection import extract_frames_from_scenes

# Extrahiere 5 Frames pro Szene
scene_frames = extract_frames_from_scenes(
    video_path='video.mp4',
    scenes=scenes,
    output_dir='frames/',
    frames_per_scene=5,
    method='uniform'  # oder 'keyframe'
)
```

**Extraction Methoden:**
- `uniform`: Gleichmäßig verteilte Frames
- `keyframe`: Erste, mittlere und letzte Frames
- `all`: Alle Frames der Szene

### 3. Szenen-Filter

```python
from src.scene_detection import filter_scenes_by_duration

# Nur Szenen zwischen 2 und 30 Sekunden
filtered_scenes = filter_scenes_by_duration(
    scenes,
    min_duration=2.0,
    max_duration=30.0
)
```

### 4. Szenen-Statistiken

```python
from src.scene_detection import get_scene_statistics

stats = get_scene_statistics(scenes)
print(f"Total scenes: {stats['total_scenes']}")
print(f"Mean duration: {stats['mean_duration']:.2f}s")
print(f"Median duration: {stats['median_duration']:.2f}s")
```

## Installation

```bash
# Aktiviere Virtual Environment
source .venv/bin/activate

# Installiere Dependencies
pip install -r requirements.txt
```

## Verwendung

### Basis: Szenen erkennen

```bash
# Szenen in Video erkennen
python src/scene_detection.py video.mp4 --output scenes.csv

# Mit verschiedenen Methoden
python src/scene_detection.py video.mp4 --method adaptive --threshold 25.0

# Mit Frame-Extraction
python src/scene_detection.py video.mp4 \
    --extract-frames \
    --frames-per-scene 5 \
    --frame-output frames/
```

### Komplett: Scene-Based Analysis Pipeline

```bash
# Vollständige Analyse mit Scene Detection
python scene_analysis_pipeline.py video.mp4 \
    --model results/test/model.joblib \
    --output results/scene_analysis \
    --scene-method content \
    --scene-threshold 27.0 \
    --min-scene-duration 1.0 \
    --frames-per-scene 5 \
    --frame-method uniform
```

Diese Pipeline:
1. **Erkennt Szenen** im Video
2. **Extrahiert Frames** aus jeder Szene
3. **Extrahiert Features** aus allen Frames
4. **Klassifiziert jede Szene** als Friend oder Foe
5. **Generiert Berichte** auf Szenen-Ebene

### Python API Beispiel

```python
from src.scene_detection import SceneDetector, extract_frames_from_scenes
from src.feature_extraction import VisualFeatureExtractor
from src.model import predict_from_features
import numpy as np

# 1. Szenen erkennen
detector = SceneDetector(method='content', threshold=27.0)
scenes = detector.detect_scenes('video.mp4', min_scene_len=1.0)

print(f"Detected {len(scenes)} scenes")
for scene in scenes[:5]:
    print(scene)

# 2. Frames extrahieren
scene_frames = extract_frames_from_scenes(
    'video.mp4',
    scenes,
    'output/frames',
    frames_per_scene=5
)

# 3. Features extrahieren und klassifizieren
extractor = VisualFeatureExtractor()

for scene_id, frame_paths in scene_frames.items():
    # Features für alle Frames der Szene
    features = []
    for frame_path in frame_paths:
        feat = extractor.extract_features(frame_path)
        features.append([feat[name] for name in extractor.feature_names])

    features = np.array(features)

    # Prediction für Szene
    predictions = predict_from_features(features, 'model.joblib')
    foe_ratio = np.mean(predictions)
    scene_label = "Foe" if foe_ratio > 0.5 else "Friend"

    print(f"Scene {scene_id}: {scene_label} (confidence: {max(foe_ratio, 1-foe_ratio):.2%})")
```

## Outputs

### Scene CSV
```csv
scene_id,start_frame,end_frame,start_time,end_time,duration,frame_count
0,0,150,0.0,6.25,6.25,150
1,151,380,6.29,15.83,9.54,229
...
```

### Scene Predictions CSV
```csv
scene_id,start_time,end_time,duration,foe_votes,friend_votes,foe_ratio,prediction,label,confidence
0,0.0,6.25,6.25,2,3,0.40,0,Friend,0.60
1,6.29,15.83,9.54,4,1,0.80,1,Foe,0.80
...
```

### Summary Report
```
SCENE-BASED FRIEND-FOE ANALYSIS REPORT
======================================================================

Video: movie.mp4
Analysis Date: 2026-02-02 14:30:00

SETTINGS
----------------------------------------------------------------------
Scene Detection Method: content
Scene Threshold: 27.0
Minimum Scene Duration: 1.0s
Frames per Scene: 5

RESULTS
----------------------------------------------------------------------
Total Scenes: 45
Friend Scenes: 28 (62.2%)
Foe Scenes: 17 (37.8%)

Total Duration: 180.5s
Friend Duration: 110.2s (61.1%)
Foe Duration: 70.3s (38.9%)

SCENE DETAILS
----------------------------------------------------------------------
Scene   0:    0.00s -    6.25s | Friend (conf: 60.00%)
Scene   1:    6.29s -   15.83s | Foe    (conf: 80.00%)
...
```

## Parameter-Tuning

### Scene Detection Threshold

**Niedrigerer Threshold** (z.B. 15-20):
- Mehr Szenen werden erkannt
- Empfindlicher auf kleine Änderungen
- Gut für: Videos mit subtilen Schnitten

**Höherer Threshold** (z.B. 30-40):
- Weniger Szenen werden erkannt
- Nur deutliche Schnitte werden erkannt
- Gut für: Videos mit vielen Kamerabewegungen

**Standard: 27.0** - Guter Kompromiss für die meisten Videos

### Frames per Scene

- **1-3 Frames**: Sehr schnell, aber weniger robust
- **5-10 Frames**: Gute Balance (empfohlen)
- **>10 Frames**: Sehr robust, aber langsamer und mehr Speicher

### Minimum Scene Duration

- **0.5s**: Erkennt auch sehr kurze Schnitte
- **1.0s**: Standard, filtert sehr kurze Szenen
- **2.0s+**: Nur längere, etablierte Szenen

## Vorteile der Szenen-basierten Analyse

1. **Reduzierter Scope**: Klassifizierung nur auf relevanten Szenen
2. **Kontextuelle Analyse**: Mehrere Frames pro Szene → robustere Predictions
3. **Zeitliche Struktur**: Verständnis der zeitlichen Abfolge
4. **Effizient**: Nicht jeder Frame muss analysiert werden
5. **Interpretierbar**: Szenen-Level Predictions sind verständlicher

## Workflow-Integration

```
Video Input
    ↓
Scene Detection → scenes.csv
    ↓
Frame Extraction → frames/scene_XXX_frame_YYY.jpg
    ↓
Feature Extraction → frame_features.csv
    ↓
Scene Classification → scene_predictions.csv
    ↓
Reports & Visualization
```

## Troubleshooting

### Problem: Zu viele/wenige Szenen erkannt

**Lösung**: Threshold anpassen
```bash
# Mehr Szenen: niedrigeren Threshold
python src/scene_detection.py video.mp4 --threshold 20.0

# Weniger Szenen: höheren Threshold
python src/scene_detection.py video.mp4 --threshold 35.0
```

### Problem: PySceneDetect nicht verfügbar

**Lösung**: Installieren oder Fallback verwenden
```bash
# Installieren
pip install scenedetect[opencv]

# Oder Fallback nutzen
python src/scene_detection.py video.mp4 --method threshold
```

### Problem: Szenen zu kurz/lang

**Lösung**: Filter anwenden
```python
filtered_scenes = filter_scenes_by_duration(
    scenes,
    min_duration=2.0,  # Mindestens 2 Sekunden
    max_duration=30.0  # Maximal 30 Sekunden
)
```

## Nächste Schritte

1. **Optical Flow Analysis**: Kamerabewegung innerhalb von Szenen
2. **Audio Analysis**: Integration von Audio-Features
3. **Temporal Models**: LSTM/Transformer für zeitliche Sequenzen
4. **Multi-Shot Tracking**: Verfolgung von Charakteren über Szenen hinweg
