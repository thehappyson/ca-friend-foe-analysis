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

**Example (4 films — recommended):**
```bash
python scripts/train_multi_film.py \
  --films jud_suess heimkehr triumph_des_willens feinde \
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

**Success threshold:** >70% mean accuracy across folds indicates systematic visual patterns that generalize across films.

---

## Current Results

| Film | Samples (us/them) | Test Accuracy | CV Mean |
|------|-------------------|---------------|---------|
| Jud Süß | 170 / 179 | 64.3% | 63.6% |
| Heimkehr | 216 / 89 | 72.1% | 65.9% |

**Top features (consistent across films):** `dof_variance`, `contrast`, `color_temperature`, `edge_density`, `texture_contrast`

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
