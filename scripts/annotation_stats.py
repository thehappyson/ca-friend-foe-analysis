"""
Annotation overview: per-film label distributions and inter-annotator agreement.

Usage:
    python annotation_overview.py

Expects CSV files produced by annotate_frames.py with columns:
    frame_path, label, confidence, notes, annotator, timestamp

Labels are: 'us', 'them', 'neutral' (exactly one per frame).

Outputs:
    - Per-film label distribution bar charts
    - Annotator comparison per film (if two annotators exist)
    - Cohen's Kappa inter-annotator agreement
    - Confusion matrix between annotators
    - Summary table printed to console
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# >>> CONFIGURATION — adjust these as needed <<<
# ============================================================

# List of (film_name, csv_path) tuples — one entry per annotator per film.
# Films with two annotators will get agreement metrics.
ANNOTATION_FILES = [
    #("Feinde", "../data/annotated/felix/feinde_annotated.csv"),
    ("Heimkehr", "../data/annotated/felix/heimkehr_annotated.csv"),
    ("Jud Süß", "../data/annotated/felix/jud_suess_annotated.csv"),
    ("Triumph des Willens", "../data/annotated/felix/triumph_des_willens_annotated.csv"),
    ("Hans Westmar", "../data/annotated/felix/westmar_annotated.csv"),
    #("Feinde", "../data/annotated/kevin/feinde_annotations.csv"),
    ("Heimkehr", "../data/annotated/kevin/heimkehr_annotations.csv"),
    ("Jud Süß", "../data/annotated/kevin/jud_suess_annotations.csv"),
    ("Triumph des Willens", "../data/annotated/kevin/triumph_des_willens_annotations.csv"),
    ("Hans Westmar", "../data/annotated/kevin/hans_westmar_annotations.csv")
]

# Annotator Mappings
#DEFAULT_ANNOTATOR_NAME = "annotator_2"
ANNOTATOR_ALIASES = {
    "Kevin" : "Annotator 1",
    "Felix" : "Annotator 2",
    "annotator_2" : "Annotator 1" #corrects a wrong assignment in the manual annotations
}
# Default label for frames with missing labels (applied in-memory only, not saved)
DEFAULT_LABEL = "other"
LABELS_TO_REPLACE = ["neutral"]

# Output directory for plots
OUTPUT_DIR = "../results/annotation_overview"

# ============================================================

LABELS = ["us", "them", "other"]
LABEL_COLORS = {"us": "#4C9BE8", "them": "#E85454", "other": "#999999"}


def load_annotations(annotation_files):
    """Load all CSVs and tag with film name."""
    records = []
    for film_name, csv_path in annotation_files:
        df = pd.read_csv(csv_path)

        # Fill missing annotator names and save back to original file
        #if df["annotator"].isna().any():
        #    df["annotator"] = df["annotator"].fillna(DEFAULT_ANNOTATOR_NAME)
        #    df.to_csv(csv_path, index=False)
        #    print(f"Filled missing annotator names in '{csv_path}' with '{DEFAULT_ANNOTATOR_NAME}'")

        # Replace annotator names with aliases
        df["annotator"] = df["annotator"].map(ANNOTATOR_ALIASES).fillna(df["annotator"])
        df["film"] = film_name

        # Fill missing labels in-memory only (not saved back)
        if df["label"].isna().any() or df["label"].any() =='neutral':
            n_missing = df["label"].isna().sum()
            df["label"] = df["label"].fillna(DEFAULT_LABEL)
            print(f"Note: filled {n_missing} missing label(s) in '{csv_path}' with '{DEFAULT_LABEL}' (in-memory only)")

        mask = df["label"].isin(LABELS_TO_REPLACE)
        if mask.any():
            print(
                f"Note: replaced {mask.sum()} '{df.loc[mask, 'label'].unique().tolist()}' label(s) in '{csv_path}' with '{DEFAULT_LABEL}' (in-memory only)")
            df.loc[mask, "label"] = DEFAULT_LABEL

        # Extract frame identifier (filename only, so annotators match on the same frame)
        df["frame_id"] = df["frame_path"].apply(lambda p: Path(p).stem)
        records.append(df)
    return pd.concat(records, ignore_index=True)


def plot_label_distribution(df, output_dir):
    """Bar chart of label distribution per film, split by annotator. One file per film."""
    for film in sorted(df["film"].unique()):
        film_df = df[df["film"] == film]
        annotators = sorted(film_df["annotator"].unique())

        fig, ax = plt.subplots(figsize=(5, 4))

        x = np.arange(len(LABELS))
        width = 0.35 if len(annotators) == 2 else 0.5

        for j, annotator in enumerate(annotators):
            ann_df = film_df[film_df["annotator"] == annotator]
            counts = [len(ann_df[ann_df["label"] == lbl]) for lbl in LABELS]
            offset = (-width / 2 + j * width) if len(annotators) == 2 else 0
            bars = ax.bar(
                x + offset, counts, width * 0.9,
                label=annotator,
                color=[LABEL_COLORS[l] for l in LABELS],
                alpha=0.9 if j == 0 else 0.55,
                edgecolor="white", linewidth=0.5,
            )
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            str(count), ha="center", va="bottom", fontsize=9)

        ax.set_title(film, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_ylabel("Frame count")
        if len(annotators) > 1:
            ax.legend(fontsize=8)

        plt.tight_layout()
        safe_film = film.replace(" ", "_").lower()
        path = Path(output_dir) / f"label_distribution_{safe_film}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_label_proportions(df, output_dir):
    """Stacked bar chart showing label proportions per film (normalized)."""
    films = sorted(df["film"].unique())
    annotators = sorted(df["annotator"].unique())

    groups = []
    group_labels = []
    for film in films:
        for ann in annotators:
            sub = df[(df["film"] == film) & (df["annotator"] == ann)]
            if len(sub) > 0:
                total = len(sub)
                proportions = {lbl: len(sub[sub["label"] == lbl]) / total for lbl in LABELS}
                proportions["group"] = f"{film}\n({ann})"
                groups.append(proportions)
                group_labels.append(f"{film}\n({ann})")

    fig, ax = plt.subplots(figsize=(max(4, 2.5 * len(groups)), 4))
    x = np.arange(len(groups))
    bottoms = np.zeros(len(groups))

    for lbl in LABELS:
        vals = [g[lbl] for g in groups]
        ax.bar(x, vals, bottom=bottoms, label=lbl, color=LABEL_COLORS[lbl],
               edgecolor="white", linewidth=0.5)
        # Add percentage labels
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_title("Label Proportions per Film / Annotator", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = Path(output_dir) / "label_proportions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def compute_agreement(df, output_dir):
    """Compute and display inter-annotator agreement per film."""
    films = sorted(df["film"].unique())
    results = []
    all_labels1 = []  # collect for pooled metrics
    all_labels2 = []

    for film in films:
        film_df = df[df["film"] == film]
        annotators = sorted(film_df["annotator"].unique())

        if len(annotators) < 2:
            print(f"\n[{film}] Only one annotator — skipping agreement.")
            continue

        a1, a2 = annotators[0], annotators[1]
        df1 = film_df[film_df["annotator"] == a1].set_index("frame_id")
        df2 = film_df[film_df["annotator"] == a2].set_index("frame_id")

        # Only compare frames annotated by both
        common = df1.index.intersection(df2.index)
        if len(common) == 0:
            print(f"\n[{film}] No overlapping frames between {a1} and {a2}.")
            continue

        labels1 = df1.loc[common, "label"].values
        labels2 = df2.loc[common, "label"].values

        # Collect for pooled computation
        all_labels1.extend(labels1)
        all_labels2.extend(labels2)

        # Cohen's Kappa
        kappa = cohen_kappa_score(labels1, labels2, labels=LABELS)

        # PABAK
        n = len(labels1)
        k = len(LABELS)
        po = np.mean(labels1 == labels2)
        pabak = (k * po - 1) / (k - 1)

        # Percent agreement
        agreement_pct = po

        # Per-label agreement
        per_label = {}
        for lbl in LABELS:
            mask = (labels1 == lbl) | (labels2 == lbl)
            if mask.sum() > 0:
                per_label[lbl] = np.mean(labels1[mask] == labels2[mask])
            else:
                per_label[lbl] = float("nan")

        results.append({
            "film": film,
            "annotator_1": a1,
            "annotator_2": a2,
            "n_common": len(common),
            "agreement_pct": agreement_pct,
            "cohens_kappa": kappa,
            "pabak": pabak,
            **{f"agreement_{lbl}": per_label[lbl] for lbl in LABELS},
        })

        # Confusion matrix
        cm = confusion_matrix(labels1, labels2, labels=LABELS)
        plot_confusion_matrix(cm, film, a1, a2, output_dir)

    if results:
        results_df = pd.DataFrame(results)

        # --- Pooled metrics (all frames across films) ---
        all_labels1 = np.array(all_labels1)
        all_labels2 = np.array(all_labels2)
        pooled_kappa = cohen_kappa_score(all_labels1, all_labels2, labels=LABELS)
        pooled_po = np.mean(all_labels1 == all_labels2)
        pooled_pabak = (len(LABELS) * pooled_po - 1) / (len(LABELS) - 1)

        pooled_per_label = {}
        for lbl in LABELS:
            mask = (all_labels1 == lbl) | (all_labels2 == lbl)
            if mask.sum() > 0:
                pooled_per_label[lbl] = np.mean(all_labels1[mask] == all_labels2[mask])
            else:
                pooled_per_label[lbl] = float("nan")

        # --- Average metrics (unweighted mean across films) ---
        avg_agreement = results_df["agreement_pct"].mean()
        avg_per_label = {lbl: results_df[f"agreement_{lbl}"].mean() for lbl in LABELS}

        # --- Print results ---
        print("\n" + "=" * 70)
        print("INTER-ANNOTATOR AGREEMENT")
        print("=" * 70)
        for _, row in results_df.iterrows():
            print(f"\n  Film: {row['film']}")
            print(f"  Annotators: {row['annotator_1']} vs {row['annotator_2']}")
            print(f"  Common frames: {row['n_common']}")
            print(f"  Agreement: {row['agreement_pct']:.1%}")
            print(f"  Cohen's Kappa: {row['cohens_kappa']:.3f}")
            print(f"  PABAK: {row['pabak']:.3f}")
            for lbl in LABELS:
                val = row[f"agreement_{lbl}"]
                print(f"    {lbl}: {val:.1%}" if not np.isnan(val) else f"    {lbl}: n/a")

        print(f"\n  {'─' * 50}")
        print(f"  POOLED (all {len(all_labels1)} frames)")
        print(f"  Agreement: {pooled_po:.1%}")
        print(f"  Cohen's Kappa: {pooled_kappa:.3f}")
        print(f"  PABAK: {pooled_pabak:.3f}")
        for lbl in LABELS:
            val = pooled_per_label[lbl]
            print(f"    {lbl}: {val:.1%}" if not np.isnan(val) else f"    {lbl}: n/a")

        print(f"\n  {'─' * 50}")
        print(f"  AVERAGE (unweighted mean over {len(results)} films)")
        print(f"  Agreement: {avg_agreement:.1%}")
        for lbl in LABELS:
            val = avg_per_label[lbl]
            print(f"    {lbl}: {val:.1%}" if not np.isnan(val) else f"    {lbl}: n/a")

        # --- Save to CSV with summary rows ---
        pooled_row = {
            "film": "POOLED",
            "annotator_1": "",
            "annotator_2": "",
            "n_common": len(all_labels1),
            "agreement_pct": pooled_po,
            "cohens_kappa": pooled_kappa,
            "pabak": pooled_pabak,
            **{f"agreement_{lbl}": pooled_per_label[lbl] for lbl in LABELS},
        }
        avg_row = {
            "film": "AVERAGE",
            "annotator_1": "",
            "annotator_2": "",
            "n_common": "",
            "agreement_pct": avg_agreement,
            "cohens_kappa": "",
            "pabak": "",
            **{f"agreement_{lbl}": avg_per_label[lbl] for lbl in LABELS},
        }
        results_df = pd.concat(
            [results_df, pd.DataFrame([pooled_row, avg_row])],
            ignore_index=True,
        )

        csv_path = Path(output_dir) / "agreement_summary.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    return results


def plot_confusion_matrix(cm, film, a1, a2, output_dir):
    """Plot a confusion matrix heatmap for one film."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel(a2, fontsize=10)
    ax.set_ylabel(a1, fontsize=10)
    ax.set_title(f"{film}\nAnnotator Confusion Matrix", fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    safe_film = film.replace(" ", "_").lower()
    path = Path(output_dir) / f"confusion_{safe_film}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary_table(df):
    """Print a console summary of all annotations."""
    print("\n" + "=" * 70)
    print("ANNOTATION SUMMARY")
    print("=" * 70)

    for film in sorted(df["film"].unique()):
        film_df = df[df["film"] == film]
        print(f"\n  {film}")
        print(f"  {'-' * 50}")

        for annotator in sorted(film_df["annotator"].unique()):
            ann_df = film_df[film_df["annotator"] == annotator]
            total = len(ann_df)
            dist = {lbl: len(ann_df[ann_df["label"] == lbl]) for lbl in LABELS}
            dist_str = " | ".join(f"{lbl}: {dist[lbl]} ({dist[lbl]/total:.0%})" for lbl in LABELS)
            print(f"    {annotator:15s} [{total:4d} frames] {dist_str}")


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(ANNOTATION_FILES)
    print(f"Loaded {len(df)} total annotations across {df['film'].nunique()} film(s).\n")

    print_summary_table(df)
    plot_label_distribution(df, output_dir)
    plot_label_proportions(df, output_dir)
    compute_agreement(df, output_dir)

    print(f"\nAll outputs saved to '{output_dir}'.")


if __name__ == "__main__":
    main()