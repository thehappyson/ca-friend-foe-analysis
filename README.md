# The Visual Construction of the Enemy

**Cultural Analytics Project – Winter Term 2025/26**
**Leipzig University**

A quantitative analysis of "Us" vs. "Them" visual representation in Nazi-era films.

---

## Research Question

Which visual features did Nazi film production systematically employ to distinguish the ingroup ("Us") from the outgroup ("Them")? Can these differences be learned automatically, and which visual features are most relevant for this distinction?

**Key Innovation:** We employ leave-one-film-out cross-validation to prove these visual strategies were systematic across Nazi propaganda, not merely film-specific artistic choices.

## Focus

The analysis focuses on the ideological and "racial" primary enemies of Nazi ideology, particularly the concept of "Jewish Bolshevism" (Jews, Communists, Soviets), rather than the depiction of Western Allies. Hitler maintained an ambivalent relationship with the British in particular, viewing them as a "brother nation."

## Methodology

### Data Collection & Annotation
- Manual frame-by-frame annotation using interactive viewer (`annotate_viewer.py`)
- 4-label system: 'us' (ingroup), 'them' (outgroup), 'other' (irrelevant), 'unlabeled'
- Target: 150-200 clean samples per film across 4 films (600-800 total samples)

### Feature Extraction
- 22 visual features across 7 categories: lighting, color, composition, texture, region properties, face detection, depth of field
- Automated extraction using OpenCV-based computer vision pipeline
- Automatic subtitle removal: crops bottom 12% of frames to eliminate subtitle confounds

### Classification & Validation
- Random Forest classifier (100 estimators, balanced class weights)
- **Leave-one-film-out cross-validation:** Train on 3 films, test on 1 held-out film
- Rotates through all 4 films to test true generalization
- Feature importance analysis to identify which visual strategies were most distinctive
- Rigorous cross-film validation proves systematic patterns vs. film-specific aesthetics

### Visual Features (22 total)

**Lighting Features (5):**
- Mean brightness, brightness std, contrast
- Low-key ratio (dark/dramatic lighting)
- High-key ratio (bright/even lighting)

**Color Features (5):**
- Saturation mean/std
- Hue mean/std
- Color temperature (warm vs. cool tones - red/(blue+green) ratio)

**Composition Features (4):**
- Edge density
- Center brightness (center region vs. overall brightness)
- Vertical symmetry (left-right mirror similarity)
- Horizontal symmetry (top-bottom mirror similarity)

**Texture Features (2):**
- Texture contrast (local variance in pixel values)
- Texture homogeneity (uniformity of texture)

**Region Features (2):**
- Dark regions count (distinct dark areas)
- Bright regions count (distinct bright areas)

**Face Detection Features (3):**
- Face count (number of faces detected)
- Face area ratio (proportion of frame occupied by faces)
- Largest face vertical position (normalized Y-coordinate of largest detected face)

**Depth of Field Proxy (1):**
- DoF variance (sharpness variance across 3×3 grid regions - higher = shallow DoF)

---

## Dataset

**Source:** [Internet Archive – German Films 1933-1945](https://archive.org/details/movies?and%5B%5D=mediatype%3A%22movies%22&and%5B%5D=language%3A%22German%22&and%5B%5D=year%3A%5B1933+TO+1945%5D)

### Sampling Strategy

- **Frame extraction rate:** 1 frame per 2 seconds (0.5 fps) to capture visual variety while maintaining manageable dataset size
- **Target:** Minimum 500 frames per category ("Us" vs "Them") for statistical validity
- **Temporal coverage:** Frames sampled across entire film duration to account for narrative progression
- **Quality filters:** Exclude transition frames (fades, cuts) and text-only title cards
- **Balance consideration:** Equal sampling from propaganda vs. entertainment films within each category

### Enemy Image Films ("Them" – Outgroup)

**Antisemitic Films:**
- *Der Ewige Jude / The Eternal Jew* (1940) – "Documentary," explicit antisemitic propaganda
- *Jud Süß / Jew Süss* (1940) – Feature film, antisemitic stereotype of the "power-hungry Jew"
- *Die Rothschilds / The Rothschilds* (1940) – Feature film, enemy image of "Jewish finance capital"

**Anti-Bolshevik/Anti-Communist Films:**
- *Hitlerjunge Quex / Hitler Youth Quex* (1933) – Communists as enemies of German youth
- *Hans Westmar* (1933) – SA martyr vs. Communists
- *GPU* (1942) – Soviet secret service as enemy image
- *Flüchtlinge / Refugees* (1933) – Volga Germans vs. Bolsheviks

**Anti-Polish/Anti-Slavic Films:**
- *Heimkehr / Homecoming* (1941) – "Ethnic Germans" as victims of Polish violence
- *Feinde / Enemies* (1940) – German minority in Poland

### Ingroup Films ("Us" – Volksgemeinschaft)

**Explicit Propaganda:**
- *Triumph des Willens / Triumph of the Will* (1935) – Nazi Party Rally, idealized "national community"
- *Olympia* (1938) – "Aryan" body ideals
- *Kolberg* (1945) – Perseverance propaganda, German heroes

**Entertainment Films with Positive Ingroup Representation:**
- *Die große Liebe / The Great Love* (1942) – German soldiers and home front
- *Wunschkonzert / Request Concert* (1940) – National community on the home front
- *Stukas* (1941) – Heroic Luftwaffe pilots

---

## Expected Outcomes

### Primary Research Goals

1. **Cross-Film Generalization** - Do visual patterns generalize to unseen films?
   - Leave-one-film-out accuracy >70% proves systematic propaganda strategies
   - Demonstrates patterns transcend individual directors, production years, target enemies
   - **Key finding:** Computational proof of unified visual propaganda doctrine

2. **Feature Importance Ranking** - Which visual strategies were most distinctive?
   - Hypothesis: "Them" portrayed with low-key lighting (dark, dramatic)
   - Hypothesis: "Us" portrayed with high-key lighting (bright, even) and centered composition
   - **Consistency analysis:** Features important across ALL folds = core propaganda toolkit

3. **Classification Performance** - Quantifying the visual distinction
   - Single-film baseline: >75% accuracy (film-specific patterns)
   - Multi-film generalization: >70% accuracy (systematic patterns)
   - Low CV variance (<20%) indicates stable, learnable strategies

4. **Historical Insights** - Quantitative evidence for propaganda theory
   - Comparison across film types (documentary vs. feature vs. rally footage)
   - Enemy-type analysis (antisemitic vs. anti-Bolshevik vs. anti-Polish)
   - Temporal consistency (early war 1933-1939 vs. late war 1940-1945)

### Potential Findings

- Systematic use of lighting contrast to dehumanize outgroup
- Camera angles and composition reinforcing power dynamics
- Color saturation differences between idealized ingroup and stigmatized outgroup
- Texture and edge density variations (chaos vs. order visual metaphors)

---

## Preliminary Results (Jud Süß - Single Film Baseline)

**Dataset:** 349 cleaned annotated frames (170 'us', 179 'them')
**Test Accuracy:** 64.3%
**Cross-validation:** 63.6% ± 10.5% (5-fold)

**Top Discriminative Features (Enhanced 22-feature set):**
1. **DoF variance** (8.6%) - Depth of field proxy (NEW)
2. **Contrast** (7.3%) - Lighting contrast differences
3. **Largest face Y-position** (7.2%) - Vertical face position (NEW)
4. **Color temperature** (6.4%) - Warm vs. cool tones (NEW)
5. **Edge density** (6.2%) - Visual complexity

**Key Findings:**
- Enhanced features (face detection, depth of field, color temperature) significantly improved performance and stability
- Cross-validation variance reduced from ±15.5% to ±10.5%, indicating more robust generalization
- Face position and depth of field features are now top predictors, validating cinematography theory
- Lighting and composition features remain critical, but enhanced features capture additional propaganda strategies

**Annotation Strategy:**
- Removed ambiguous mixed frames (both us/them characters visible)
- Excluded corrupted German characters (collaborators labeled as 'other', not 'them')
- Focus on racial/ideological enemy coding, not general morality

---

## Quick Start (POC)

### Setup Virtual Environment (Recommended)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # Linux/Mac
# OR on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Test the Pipeline

```bash
# 1. Run the complete pipeline test with synthetic data
python test_pipeline.py

# 2. Or try the interactive notebook
jupyter notebook notebooks/00_quick_start_demo.ipynb
```

See [USAGE.md](USAGE.md) for detailed instructions on using real video data.

---

## Multi-Film Research Workflow

### Recommended Approach for Publication-Quality Results

**Goal:** 4 films × 200 samples = 800 total annotated frames

**Film Selection (Balanced):**
- **"Them" Films:** Jud Süß, Der Ewige Jude (antisemitic)
- **"Us" Films:** Triumph des Willens, Kolberg (nationalist/heroic)

**Workflow:**

1. **Extract frames for all films:**
   ```bash
   for film in jud_suess der_ewige_jude triumph_des_willens kolberg; do
     python src/frame_extraction.py data/videos/${film}.mp4 --output frames/${film}
   done
   ```

2. **Annotate each film (target 150-200 clean samples per film):**
   ```bash
   python annotate_viewer.py frames/jud_suess data/annotated/jud_suess_annotations.csv
   # Repeat for other films
   ```

3. **Extract features for all annotated films:**
   ```bash
   for film in jud_suess der_ewige_jude triumph_des_willens kolberg; do
     python src/feature_extraction.py frames/${film} data/features/${film}_features.csv
   done
   ```

4. **Run leave-one-film-out cross-validation:**
   ```bash
   python train_multi_film.py \
     --films jud_suess der_ewige_jude triumph_des_willens kolberg \
     --output results/multi_film
   ```

**Expected Outputs:**
- Per-film test accuracies
- Average generalization accuracy (target: >70%)
- Feature importance consistency across all folds
- Confusion matrices for each fold
- Publication-ready visualizations

**Key Insight:** If accuracy remains >70% across all folds, this proves Nazi propaganda employed systematic visual strategies across different directors, production years, and enemy targets.

### Pipeline Overview

1. **Frame Extraction** - Extract frames from videos at regular intervals
2. **Interactive Annotation** - Use keyboard-driven viewer to label frames efficiently
   ```bash
   python annotate_viewer.py frames/film_name data/annotated/film_name_annotations.csv
   ```
3. **Feature Extraction** - Extract 22 visual features (lighting, color, composition, texture, faces, depth of field)
   ```bash
   python src/feature_extraction.py frames/film_name data/features/film_name_features.csv
   ```
4. **Multi-Film Training** - Leave-one-film-out cross-validation for rigorous testing
   ```bash
   python train_multi_film.py --films film1 film2 film3 film4 --output results/multi_film
   ```
5. **Analysis** - Feature importance, confusion matrices, per-film accuracies, generalization metrics

---

## Project Structure

```
├── data/
│   ├── videos/             # Original video files (MP4)
│   ├── annotated/          # CSV files with frame annotations
│   └── features/           # CSV files with extracted features
├── frames/                 # Extracted frames organized by film
│   ├── jud_suess/
│   ├── der_ewige_jude/
│   ├── triumph_des_willens/
│   └── kolberg/
├── notebooks/
│   └── 00_quick_start_demo.ipynb    # Interactive demo notebook
├── src/
│   ├── frame_extraction.py    # Video → frames extraction
│   ├── annotation.py          # Annotation tools and CSV management
│   ├── feature_extraction.py  # Visual feature extraction (22 features + subtitle removal)
│   ├── model.py              # Single-film Random Forest classifier
│   ├── scene_detection.py    # Scene/shot detection utilities
│   ├── face_analysis.py      # Face detection and analysis
│   ├── generate_test_data.py # Synthetic test data generator
│   └── utils.py              # Utility functions
├── annotate_viewer.py         # Interactive annotation viewer (OpenCV)
├── train_multi_film.py        # Leave-one-film-out cross-validation
├── visualize_feature_space.py # PCA/t-SNE feature space visualization
├── results/
│   ├── jud_suess/            # Single-film baseline results
│   ├── multi_film/           # Multi-film CV results
│   │   ├── per_fold_results.csv
│   │   ├── feature_importance_consistency.csv
│   │   └── per_film_accuracies.png
│   └── figures/              # Additional visualizations
├── test_pipeline.py          # End-to-end pipeline test script
├── README.md
├── CLAUDE.md                 # Developer/AI assistant guidance
├── USAGE.md                  # Detailed usage instructions
└── requirements.txt
```

---

## Authors

| | Kevin Kunkel | Felix Filius |
|---|---|---|
| **Program** | M.Sc. Computer Science | M.Sc. Data Science |
| **E-Mail** | mm21lugi@studserv.uni-leipzig.de | zq25ohog@studserv.uni-leipzig.de |
| **Student ID** | 3738957 | 3773660 |

---

## License

This project was created as part of the "Cultural Analytics" module at Leipzig University.

## Disclaimer

The films analyzed in this project contain National Socialist propaganda and antisemitic content. This analysis serves exclusively academic purposes in the context of historical research and education.