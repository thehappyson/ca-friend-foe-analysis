# Quick Start Guide

This POC implementation provides tools for analyzing visual differences between "Us" (ingroup) and "Them" (outgroup) representations in Nazi propaganda films.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Project structure:**
```
ca-friend-foe-analysis/
├── data/
│   ├── raw/              # Extracted frames
│   ├── annotated/        # Annotation CSV files
│   └── features/         # Extracted feature CSV files
├── notebooks/
│   └── 00_quick_start_demo.ipynb  # Interactive demo
├── src/
│   ├── frame_extraction.py    # Extract frames from videos
│   ├── annotation.py          # Annotation tools
│   ├── feature_extraction.py  # Visual feature extraction
│   ├── model.py              # ML classifier
│   └── utils.py              # Utility functions
└── results/
    ├── figures/          # Generated plots
    └── tables/           # Results tables
```

## Quick Start Workflow

### Option 1: Using Jupyter Notebook (Recommended for beginners)

```bash
jupyter notebook notebooks/00_quick_start_demo.ipynb
```

Follow the notebook step-by-step.

### Option 2: Using Command Line

#### 1. Extract frames from video

```bash
# Extract 100 uniformly distributed frames
python src/frame_extraction.py path/to/video.mp4 \
    --output data/raw/film_name \
    --uniform 100

# Or extract frames every 1 second
python src/frame_extraction.py path/to/video.mp4 \
    --output data/raw/film_name \
    --interval 1.0 \
    --max-frames 200
```

#### 2. Create annotation template

```bash
python src/annotation.py create-template \
    data/raw/film_name \
    data/annotated/film_name_annotations.csv
```

#### 3. Annotate frames

Open `data/annotated/film_name_annotations.csv` in Excel/LibreOffice and fill in the `label` column:
- `us` - Ingroup (idealized Germans, heroes)
- `them` - Outgroup (Jews, Communists, enemies)
- `neutral` - Neither
- `unclear` - Can't determine

**You need at least 20-30 frames labeled as 'us' and 'them' for training.**

#### 4. Extract features

```bash
python src/feature_extraction.py \
    data/raw/film_name \
    data/features/film_name_features.csv
```

#### 5. Train classifier

```bash
python src/model.py \
    data/features/film_name_features.csv \
    data/annotated/film_name_annotations.csv \
    results/
```

This will:
- Train a Random Forest classifier
- Evaluate on test set
- Perform cross-validation
- Generate feature importance plots
- Save results to `results/`

## Understanding the Results

### Feature Importance

The most important features reveal which visual characteristics distinguish the groups.

**Key features to look for:**

**Lighting:**
- `low_key_ratio` - Proportion of dark pixels (dramatic, sinister)
- `high_key_ratio` - Proportion of bright pixels (heroic, clean)
- `contrast` - Overall contrast (dramatic vs soft)

**Composition:**
- `vertical_symmetry` - Symmetrical = ordered, powerful
- `center_brightness` - Center focus = important subject
- `edge_density` - Edges/lines = busy vs clean

**Color:**
- `saturation_mean` - Vibrant vs desaturated
- `hue_mean` - Warm (red/yellow) vs cool (blue/green)

### Expected Patterns (based on propaganda research)

**"Us" (Ingroup) typically shows:**
- Bright, even lighting (high-key)
- Symmetrical, ordered compositions
- Clean framing
- Warm or neutral colors

**"Them" (Outgroup) typically shows:**
- Dark, shadowy lighting (low-key)
- Asymmetrical, chaotic compositions
- Cluttered framing
- Desaturated or cool colors

## Minimal Example

For a quick test with 2 videos:

```bash
# Extract frames from "Us" film
python src/frame_extraction.py triumph_des_willens.mp4 --output data/raw/us_film --uniform 50

# Extract frames from "Them" film
python src/frame_extraction.py der_ewige_jude.mp4 --output data/raw/them_film --uniform 50

# Create annotation template
python src/annotation.py create-template data/raw/us_film data/annotated/us_annotations.csv
python src/annotation.py create-template data/raw/them_film data/annotated/them_annotations.csv

# Manually label frames in the CSV files
# Then merge annotations:
cat data/annotated/us_annotations.csv data/annotated/them_annotations.csv > data/annotated/combined_annotations.csv

# Extract features
python src/feature_extraction.py data/raw/us_film data/features/us_features.csv
python src/feature_extraction.py data/raw/them_film data/features/them_features.csv

# Merge features (in Python):
# pd.concat([pd.read_csv('data/features/us_features.csv'),
#            pd.read_csv('data/features/them_features.csv')]).to_csv('data/features/combined_features.csv', index=False)

# Train model
python src/model.py data/features/combined_features.csv data/annotated/combined_annotations.csv results/
```

## Tips for Success

1. **Start small**: 2 films, 50-100 frames each
2. **Clear contrast**: Choose films with obvious visual differences
3. **Consistent annotation**: Define clear rules for labeling
4. **Balance classes**: Try to have similar numbers of 'us' and 'them' frames
5. **Document everything**: Keep notes on which films, which scenes, annotation criteria

## Troubleshooting

**"No labeled data found"**
- Check that your CSV has 'us' and 'them' labels (not empty)
- Verify frame_path column matches actual file paths

**"Insufficient labeled data"**
- Need at least 10 samples per class, ideally 30+
- Annotate more frames

**Low accuracy**
- Try more training data
- Check if visual differences are actually present
- Consider using different/additional features

**"Could not load image"**
- Check file paths are correct
- Verify images exist in data/raw/

## Next Steps

After validating the POC:

1. Expand to more films
2. Add more sophisticated features (face detection, pose estimation)
3. Try deep learning features (VGG, ResNet)
4. Perform qualitative analysis of results
5. Write up findings in paper format (see Hinweise Projektarbeit.pdf)
