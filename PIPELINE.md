# Training Pipeline Documentation

Step-by-step guide to run the full pipeline from frames to model evaluation.

---

## Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 1 — Extract Frames

Extract frames from a video file into `frames/<film_name>/`.

```bash
python scripts/extract_frames.py data/videos/<film_name>.mp4 data/shots/<film_name>.mp4.scenes.txt
```

> Output: PNG frames saved to `frames/<film_name>/`

---

## Step 2 — Annotate Frames

Launch the interactive annotation viewer. Creates a new CSV if none exists, or resumes from the first unlabeled frame.

```bash
python scripts/annotate_viewer.py frames/<film_name> data/annotated/kevin/<film_name>_annotations.csv
```

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `1` or `U` | Label as **us** (ingroup) |
| `2` or `T` | Label as **them** (outgroup) |
| `3` or `O` | Label as **other** (skip) |
| `0` | Clear label (unlabeled) |
| `→` / `D` | Next frame |
| `←` / `A` | Previous frame |
| `S` | Save |
| `Q` | Quit and save |

**Label guidelines:**
- `us` — Ingroup character clearly dominant in frame (>70% screen space), positive visual treatment
- `them` — Outgroup character clearly dominant in frame, negative visual treatment
- `other` — Landscapes, title cards, crowd shots, mixed frames, ambiguous content, when uncertain

Auto-saves every 10 labels. Target **150–200 clean `us`/`them` labels per film**.

---

## Step 3 — Extract Visual Features

Extracts 22 visual features per frame (lighting, color, composition, texture, face detection, depth-of-field).

```bash
python src/feature_extraction.py frames/<film_name> data/features/<film_name>_features.csv
```

> Output: `data/features/<film_name>_features.csv`

---

## Step 4a — Single-Film Baseline

Train and evaluate a classifier on one film. Uses 80/20 stratified train/test split + 5-fold cross-validation.

```bash
python src/model.py \
  data/features/<film_name>_features.csv \
  data/annotated/kevin/<film_name>_annotations.csv \
  results/<film_name>
```

**Example (Jud Süß):**
```bash
python src/model.py \
  data/features/jud_suess_features.csv \
  data/annotated/kevin/jud_suess_annotations.csv \
  results/jud_suess
```

**Example (Heimkehr):**
```bash
python src/model.py \
  data/features/heimkehr_features.csv \
  data/annotated/kevin/heimkehr_annotations.csv \
  results/heimkehr
```

> Output in `results/<film_name>/`:
> - `model.joblib` — trained model
> - `evaluation_metrics.csv` — accuracy, precision, recall, F1
> - `feature_importance.csv` + `feature_importance.png`
> - `confusion_matrix.png`

**Success threshold:** >75% accuracy indicates the classifier works for this film.

---

## Step 4b — Multi-Film Leave-One-Movie-Out Cross-Validation

Tests whether visual propaganda patterns **generalize across films** — the key research validation. Trains on N−1 films, tests on the held-out film, rotates through all films.

Requires features + annotations for **all films** listed. Run Steps 1–3 for each film first.

```bash
python scripts/train_multi_film.py \
  --films <film1> <film2> <film3> ... \
  --features-dir data/features \
  --annotations-dir data/annotated/kevin \
  --output results/multi_film
```

**Example (2 films — minimal):**
```bash
python scripts/train_multi_film.py \
  --films jud_suess heimkehr \
  --features-dir data/features \
  --annotations-dir data/annotated/kevin \
  --output results/multi_film
```

**Example (3 films — final corpus):**
```bash
python scripts/train_multi_film.py \
  --films jud_suess heimkehr hans_westmar \
  --features-dir data/features \
  --annotations-dir data/annotated/kevin \
  --output results/multi_film
```

> Output in `results/multi_film/`:
> - `per_fold_results.csv` — accuracy per film
> - `summary_statistics.csv` — mean/std across all folds
> - `feature_importance_consistency.csv` — which features are consistently important
> - `per_film_accuracies.png`
> - `feature_importance_consistency.png`
> - `fold_N_<film>/model.joblib` — per-fold trained models

**Interpretation:** Results near chance level (~50%) indicate film-specific rather than systematic cross-film visual patterns. Results clearly above chance (>60%) would suggest generalizable propaganda strategies.

---

## Current Results

| Film | Samples (us/them) | Test Accuracy | Balanced CV-Acc. (5-fold) | CV Std. |
|------|-------------------|---------------|---------------------------|---------|
| *Jud Süß* (1940) | 171 / 178 | 74.3% | 62.4% | ±5.1% |
| *Heimkehr* (1941) | 216 / 89 | 62.5% | 56.4% | ±10.9% |
| *Hans Westmar* (1933) | 105 / 113 | 79.6% | 59.7% | ±11.0% |

**LOMOCV (3 films):** 54.0% ± 5.9% balanced accuracy — near chance level, indicating film-specific rather than cross-film generalizable patterns.

**Top features (consistent across LOMOCV folds):** `dof_variance`, `contrast`, `mean_brightness`, `edge_density`, `texture_contrast`

---

## Data Naming Convention

```
frames/<film_name>/                          ← extracted PNGs
data/features/<film_name>_features.csv       ← visual features
data/annotated/kevin/<film_name>_annotations.csv   ← labels
results/<film_name>/                         ← single-film results
results/multi_film/                          ← LOMO results
```

Film names used in the project: `jud_suess`, `heimkehr`, `triumph_des_willens`, `hans_westmar`,
