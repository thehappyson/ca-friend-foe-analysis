# The Visual Construction of the Enemy

**Cultural Analytics Project – Winter Term 2025/26**
**Leipzig University**

A quantitative analysis of "Us" vs. "Them" visual representation in Nazi-era films.

---

## Research Question

Which visual features did Nazi film production systematically employ to distinguish the ingroup ("Us") from the outgroup ("Them")? Can these differences be learned automatically, and which visual features are most relevant for this distinction?

## Focus

The analysis focuses on the ideological and "racial" primary enemies of Nazi ideology, particularly the concept of "Jewish Bolshevism" (Jews, Communists, Soviets), rather than the depiction of Western Allies. Hitler maintained an ambivalent relationship with the British in particular, viewing them as a "brother nation."

## Methodology

- Extraction of visual features (lighting, camera angle, shot scale, composition, pose) from annotated frames
- Training of a classical ML classifier (e.g., Random Forest) to distinguish "Us" from "Them"
- Feature importance analysis to identify which visual characteristics propaganda most strongly employed for enemy image construction
- Cross-validation strategy with stratified k-fold to ensure robust results
- Inter-annotator agreement metrics to validate manual frame labeling
- Class imbalance handling through stratified sampling and balanced class weights

### Visual Features (17 total)

**Lighting Features (5):**
- Mean brightness, brightness std, contrast
- Low-key ratio (dark/dramatic lighting)
- High-key ratio (bright/even lighting)

**Color Features (4):**
- Saturation mean/std
- Hue mean/std

**Composition Features (4):**
- Edge density
- Center brightness
- Vertical/horizontal symmetry

**Texture Features (2):**
- Texture contrast
- Texture homogeneity

**Region Features (2):**
- Dark regions count
- Bright regions count

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

1. **Feature Importance Ranking** - Which visual strategies were most distinctive for enemy construction?
   - Hypothesis: "Them" portrayed with low-key lighting (dark, dramatic)
   - Hypothesis: "Us" portrayed with high-key lighting (bright, even) and centered composition

2. **Classification Performance** - How "learnable" is the visual distinction?
   - Target: >70% accuracy indicates systematic visual differentiation
   - Feature ablation studies to test individual feature category contributions

3. **Historical Insights** - Quantitative evidence for propaganda strategies
   - Comparison across film types (documentary vs. feature films)
   - Temporal analysis (early war 1933-1939 vs. late war 1940-1945)
   - Identification of consistent vs. context-dependent visual patterns

### Potential Findings

- Systematic use of lighting contrast to dehumanize outgroup
- Camera angles and composition reinforcing power dynamics
- Color saturation differences between idealized ingroup and stigmatized outgroup
- Texture and edge density variations (chaos vs. order visual metaphors)

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

### Pipeline Overview

1. **Frame Extraction** - Extract frames from videos at regular intervals
2. **Annotation** - Label frames as 'us' (ingroup) or 'them' (outgroup)
3. **Feature Extraction** - Extract 17 visual features (lighting, composition, color, texture)
4. **Classification** - Train Random Forest classifier to distinguish groups
5. **Analysis** - Identify most important visual characteristics

---

## Project Structure

```
├── data/
│   ├── videos/             # Original video files (MP4)
│   ├── raw/                # Extracted frames from videos (JPG)
│   ├── annotated/          # CSV files with frame annotations
│   └── features/           # CSV files with extracted features
├── notebooks/
│   └── 00_quick_start_demo.ipynb    # Interactive demo notebook
├── src/
│   ├── frame_extraction.py    # Video → frames extraction
│   ├── annotation.py          # Annotation tools and CSV management
│   ├── feature_extraction.py  # Visual feature extraction (17 features)
│   ├── model.py              # Random Forest classifier + evaluation
│   ├── generate_test_data.py # Synthetic test data generator
│   └── utils.py              # Utility functions
├── results/
│   ├── figures/          # Generated plots (feature importance, etc.)
│   └── tables/           # Results tables
├── test_pipeline.py      # End-to-end pipeline test script
├── README.md
├── USAGE.md              # Detailed usage instructions
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