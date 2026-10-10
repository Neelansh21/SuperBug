#!/usr/bin/env python3
"""

====================================================

This script performs rigorous EDA on the DRIAMS-A MALDI-TOF
mass spectrometry dataset, specifically targeting our 3 pathogen–antibiotic
pairs defined in Phase 0.

Generated outputs
-----------------
Figures  → notebooks/figures/
    01_class_distribution.png          Class balance bar charts
    02_mean_spectra_overlay.png        Mean S-vs-R spectra per target
    03_variance_distribution.png       Per-bin variance (log-scale)
    04_correlation_*.png               Correlation heatmap (top-50 bins)
    05_pca_variance_*.png              Explained-variance + cumulative curve
    06_pca_2d_*.png                    2-D PCA scatter (S vs R)

Processed data → data/processed/
    {species}_{antibiotic}_X_processed.npy   PCA-transformed features
    {species}_{antibiotic}_X_tic.npy         TIC-normalised spectra
    {species}_{antibiotic}_y.npy             Binary labels
    {species}_{antibiotic}_metadata.csv      Aligned metadata
    {species}_{antibiotic}_pipeline.joblib   Fitted sklearn pipeline

Usage
-----
    python notebooks/eda_driams.py

"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend so the script can run headless
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# ───────────────────────────────────────────────────────────────────────────
# Path setup — add the project root to sys.path so that this script
# can import from ml_pipeline regardless of the working directory.
# ───────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline.preprocessing import (
    load_metadata,
    filter_target_species,
    binarize_labels,
    load_spectra_batch,
    tic_normalize,
    build_preprocessing_pipeline,
    TARGET_CONFIG,
    TARGET_SPECIES,
    N_BINS,
)

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DRIAMS_ROOT  = os.path.join(PROJECT_ROOT, "DRIAMS-A")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "notebooks", "figures")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────
# Aesthetic palette — chosen a dark-mode colour scheme because it
# provides better contrast for overlaid spectra and scatter plots.
# ───────────────────────────────────────────────────────────────────────────
COLORS = {
    "susceptible": "#2ecc71",  # green
    "resistant":   "#e74c3c",  # red
    "neutral":     "#3498db",  # blue
    "accent":      "#f39c12",  # orange
    "bg":          "#1a1a2e",  # dark background
    "grid":        "#30304a",  # grid lines
    "text":        "#e0e0e0",  # light text
}

SPECIES_COLORS = {
    "Staphylococcus aureus":    "#e74c3c",
    "Klebsiella pneumoniae":    "#3498db",
    "Escherichia coli":         "#2ecc71",
}


def setup_plot_style():
    """Apply a dark, modern Matplotlib style globally."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":   COLORS["bg"],
        "axes.facecolor":     COLORS["bg"],
        "axes.edgecolor":     COLORS["grid"],
        "axes.grid":          True,
        "grid.color":         COLORS["grid"],
        "grid.alpha":         0.3,
        "text.color":         COLORS["text"],
        "axes.labelcolor":    COLORS["text"],
        "xtick.color":        COLORS["text"],
        "ytick.color":        COLORS["text"],
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  COLORS["bg"],
    })


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════

def section_metadata_overview(metadata: pd.DataFrame):
    """Print a comprehensive summary of the full DRIAMS-A metadata."""
    print("\n" + "=" * 70)
    print("  SECTION 1 :  DRIAMS-A METADATA OVERVIEW")
    print("=" * 70)

    print(f"\n  Total samples (all species):  {len(metadata):>10,}")
    print(f"  Unique species:               {metadata['species'].nunique():>10,}")
    print(f"  Years covered:                {sorted(metadata['year'].unique())}")
    print(f"  Metadata columns:             {len(metadata.columns):>10,}")

    # — Samples per year —
    print("\n  Samples per year:")
    for year in sorted(metadata["year"].unique()):
        count = (metadata["year"] == year).sum()
        print(f"    {year}: {count:>8,}")

    # — Top 15 species —
    print("\n  Top 15 species by sample count:")
    top = metadata["species"].value_counts().head(15)
    for sp, count in top.items():
        marker = "  ◀ TARGET" if sp in TARGET_SPECIES else ""
        print(f"    {sp:45s} {count:>6,}{marker}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Target Species Filtering
# ═══════════════════════════════════════════════════════════════════════════

def section_target_filtering(metadata: pd.DataFrame) -> pd.DataFrame:
    """Filter for the 3 target pathogens and report per-year breakdown."""
    print("\n" + "=" * 70)
    print("  SECTION 2 :  TARGET SPECIES FILTERING")
    print("=" * 70)

    target_df = filter_target_species(metadata)

    # Per-species per-year cross-table
    print("\n  Per-species × per-year breakdown:")
    pivot = (
        target_df
        .groupby(["species", "year"])
        .size()
        .unstack(fill_value=0)
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    print(pivot.to_string())

    return target_df


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Label Binarization & Class Distribution
# ═══════════════════════════════════════════════════════════════════════════

def section_class_distribution(target_df: pd.DataFrame) -> dict:
    """Binarize labels for each target and plot class-balance bar charts."""
    print("\n" + "=" * 70)
    print("  SECTION 3 :  LABEL BINARIZATION  &  CLASS DISTRIBUTION")
    print("=" * 70)

    results = {}

    n_targets = len(TARGET_CONFIG)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    for i, (species, config) in enumerate(TARGET_CONFIG.items()):
        antibiotic = config["primary"]
        print(f"\n  ▶ {species}  —  {antibiotic}")

        species_df = target_df[target_df["species"] == species].copy()
        labeled_df = binarize_labels(species_df, antibiotic)
        results[(species, antibiotic)] = labeled_df

        # ── bar chart ──
        ax = axes[i]
        n_s = (labeled_df["label"] == 0).sum()
        n_r = (labeled_df["label"] == 1).sum()

        bars = ax.bar(
            ["Susceptible", "Resistant"],
            [n_s, n_r],
            color=[COLORS["susceptible"], COLORS["resistant"]],
            edgecolor="white",
            linewidth=0.5,
            width=0.6,
        )

        # Add count + percentage labels directly on each bar for clarity.
        total = n_s + n_r
        for bar, val in zip(bars, [n_s, n_r]):
            pct = val / total * 100 if total else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=COLORS["text"],
            )

        short_species = species.split()[-1]
        ax.set_title(
            f"{short_species}\n({antibiotic})",
            fontsize=11, fontweight="bold",
        )
        ax.set_ylabel("Sample Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Class Distribution: Susceptible vs Resistant",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "01_class_distribution.png")
    plt.savefig(out)
    plt.close()
    print(f"\n  ✓ Saved: {out}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTIONS 4–6 — Spectra Loading, TIC Normalization, Mean Spectra Overlay
# ═══════════════════════════════════════════════════════════════════════════

def section_spectra_analysis(labeled_results: dict) -> dict:
    """Load spectra, TIC-normalize, and generate mean-spectra overlay plots."""
    print("\n" + "=" * 70)
    print("  SECTIONS 4-6 :  SPECTRA · TIC NORM · MEAN SPECTRA")
    print("=" * 70)

    spectra_data = {}

    for (species, antibiotic), labeled_df in labeled_results.items():
        print(f"\n  ▶ Loading spectra for {species} / {antibiotic} …")

        # 4 — load
        X_raw, valid_idx = load_spectra_batch(labeled_df, DRIAMS_ROOT)
        aligned_df = labeled_df.loc[valid_idx].reset_index(drop=True)
        y = aligned_df["label"].values

        # 5 — safety imputation
        X_raw = np.nan_to_num(X_raw, nan=0.0)

        # 6 — TIC normalize
        X_tic = tic_normalize(X_raw)

        print(f"    Matrix shape:   {X_raw.shape}")
        print(f"    Labels:         S = {(y == 0).sum():,}   R = {(y == 1).sum():,}")

        spectra_data[(species, antibiotic)] = {
            "X_raw": X_raw,
            "X_tic": X_tic,
            "y": y,
            "metadata": aligned_df,
        }

    # ── Mean Spectra Overlay ──
    n_pairs = len(spectra_data)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 5 * n_pairs))
    if n_pairs == 1:
        axes = [axes]

    bin_indices = np.arange(N_BINS)

    for i, ((species, antibiotic), data) in enumerate(spectra_data.items()):
        ax = axes[i]
        X_tic = data["X_tic"]
        y = data["y"]

        mean_s = X_tic[y == 0].mean(axis=0)
        mean_r = X_tic[y == 1].mean(axis=0)

        # Plot both classes as semi-transparent lines so divergent peaks
        # are clearly visible, and shade the gap between them in amber so
        # the clinician can immediately spot discriminative m/z regions.
        ax.plot(
            bin_indices, mean_s,
            color=COLORS["susceptible"], alpha=0.85, linewidth=0.6,
            label="Susceptible (S)",
        )
        ax.plot(
            bin_indices, mean_r,
            color=COLORS["resistant"], alpha=0.85, linewidth=0.6,
            label="Resistant (R)",
        )
        ax.fill_between(
            bin_indices, mean_s, mean_r,
            alpha=0.15, color=COLORS["accent"],
        )

        ax.set_title(
            f"{species}  —  {antibiotic}   (Mean Spectrum: S vs R)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("m/z Bin Index  (0 – 5 999)")
        ax.set_ylabel("TIC-Normalised Intensity")
        ax.legend(loc="upper right", framealpha=0.7)
        ax.set_xlim(0, N_BINS)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "02_mean_spectra_overlay.png")
    plt.savefig(out)
    plt.close()
    print(f"\n  ✓ Saved: {out}")

    return spectra_data


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Variance Analysis & Feature Selection
# ═══════════════════════════════════════════════════════════════════════════

def section_variance_analysis(spectra_data: dict) -> dict:
    """Compute per-bin variance and plot the distribution (log scale)."""
    print("\n" + "=" * 70)
    print("  SECTION 7 :  VARIANCE ANALYSIS  &  FEATURE SELECTION")
    print("=" * 70)

    n = len(spectra_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    variance_info = {}

    for i, ((species, antibiotic), data) in enumerate(spectra_data.items()):
        X_tic = data["X_tic"]
        variances = np.var(X_tic, axis=0)

        # Apply a VarianceThreshold here to quantify how many near-zero-
        # variance bins exist.  These correspond to baseline noise regions
        # of the MALDI-TOF spectrum that carry no biological signal.
        vt = VarianceThreshold(threshold=1e-10)
        vt.fit(X_tic)
        n_retained = vt.get_support().sum()
        n_removed = N_BINS - n_retained

        print(f"\n  ▶ {species}  /  {antibiotic}")
        print(f"    Features retained:  {n_retained:,} / {N_BINS:,}")
        print(f"    Features removed:   {n_removed:,}")
        print(f"    Max variance:       {variances.max():.6e}")
        print(f"    Median variance:    {np.median(variances):.6e}")

        variance_info[(species, antibiotic)] = {
            "variances": variances,
            "n_retained": n_retained,
            "support": vt.get_support(),
        }

        # ── plot ──
        ax = axes[i]
        sorted_var = np.sort(variances)[::-1]
        ax.semilogy(
            np.arange(N_BINS), sorted_var,
            color=SPECIES_COLORS.get(species, COLORS["neutral"]),
            linewidth=0.8,
        )
        ax.axhline(
            y=1e-10, color=COLORS["accent"], linestyle="--",
            linewidth=1, alpha=0.7, label="Threshold (1e-10)",
        )
        short = species.split()[-1]
        ax.set_title(f"{short} — Feature Variance", fontsize=11, fontweight="bold")
        ax.set_xlabel("Feature Rank")
        ax.set_ylabel("Variance (log scale)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "03_variance_distribution.png")
    plt.savefig(out)
    plt.close()
    print(f"\n  ✓ Saved: {out}")

    return variance_info


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Correlation Analysis (Top-Variance Features)
# ═══════════════════════════════════════════════════════════════════════════

def section_correlation_analysis(
    spectra_data: dict,
    variance_info: dict,
    top_n: int = 50,
):
    """
    Compute and plot correlation matrices on the *top_n* highest-variance
    features.

    Limit the correlation analysis to the top 50 bins because
    computing a full 6 000 × 6 000 matrix would be ~36 M pairwise
    calculations, which is wasteful given that most bins are noise.
    """
    print("\n" + "=" * 70)
    print("  SECTION 8 :  CORRELATION ANALYSIS  (Top-Variance Features)")
    print("=" * 70)

    for (species, antibiotic), data in spectra_data.items():
        X_tic = data["X_tic"]
        variances = variance_info[(species, antibiotic)]["variances"]

        top_idx = np.argsort(variances)[-top_n:]
        X_top = X_tic[:, top_idx]

        corr_matrix = np.corrcoef(X_top.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")

        ax.set_title(
            f"{species}  —  Correlation Matrix\n"
            f"(Top {top_n} High-Variance m/z Bins)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Feature Index")

        plt.tight_layout()
        safe_name = species.replace(" ", "_").lower()
        out = os.path.join(FIGURES_DIR, f"04_correlation_{safe_name}.png")
        plt.savefig(out)
        plt.close()
        print(f"  ✓ Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9 — PCA
# ═══════════════════════════════════════════════════════════════════════════

def section_pca_analysis(spectra_data: dict) -> dict:
    """
    Apply the full preprocessing pipeline (Variance → Scaler → PCA) and
    generate explained-variance curves + 2-D projections.
    """
    print("\n" + "=" * 70)
    print("  SECTION 9 :  PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 70)

    pca_results = {}

    for (species, antibiotic), data in spectra_data.items():
        X_tic = data["X_tic"]
        y     = data["y"]

        pipeline = build_preprocessing_pipeline(
            variance_threshold=1e-10,
            n_components=0.95,
        )
        X_pca = pipeline.fit_transform(X_tic)

        pca_step = pipeline.named_steps["pca"]
        vf_step  = pipeline.named_steps["variance_filter"]

        n_components = X_pca.shape[1]
        explained    = pca_step.explained_variance_ratio_
        cumulative   = np.cumsum(explained)

        print(f"\n  ▶ {species}  /  {antibiotic}")
        print(f"    Features after variance filter:  {vf_step.get_support().sum()}")
        print(f"    PCA components (95 % var):       {n_components}")
        print(f"    Total variance explained:        {cumulative[-1]:.2%}")

        pca_results[(species, antibiotic)] = {
            "X_pca":      X_pca,
            "pipeline":   pipeline,
            "explained":  explained,
            "cumulative": cumulative,
        }

        safe_name = species.replace(" ", "_").lower()

        # ── Fig A: Explained variance (individual + cumulative) ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        sp_color = SPECIES_COLORS.get(species, COLORS["neutral"])

        ax1.bar(
            range(min(30, n_components)),
            explained[:30],
            color=sp_color, alpha=0.8,
        )
        ax1.set_title("Individual Explained Variance", fontweight="bold")
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Variance Ratio")

        ax2.plot(
            range(n_components), cumulative,
            color=sp_color, linewidth=2,
        )
        ax2.axhline(
            y=0.95, color=COLORS["accent"],
            linestyle="--", linewidth=1,
            label="95 % threshold",
        )
        ax2.set_title("Cumulative Explained Variance", fontweight="bold")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Variance")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=9)

        fig.suptitle(
            f"{species}  —  PCA Analysis",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        out1 = os.path.join(FIGURES_DIR, f"05_pca_variance_{safe_name}.png")
        plt.savefig(out1)
        plt.close()

        # ── Fig B: 2-D PCA scatter ──
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            X_pca[y == 0, 0], X_pca[y == 0, 1],
            c=COLORS["susceptible"], alpha=0.4, s=10,
            label="Susceptible",
        )
        ax.scatter(
            X_pca[y == 1, 0], X_pca[y == 1, 1],
            c=COLORS["resistant"], alpha=0.4, s=10,
            label="Resistant",
        )
        ax.set_xlabel(f"PC1  ({explained[0]:.1%} variance)")
        ax.set_ylabel(f"PC2  ({explained[1]:.1%} variance)")
        ax.set_title(
            f"{species}  —  PCA 2-D Projection",
            fontweight="bold",
        )
        ax.legend(framealpha=0.7)
        plt.tight_layout()
        out2 = os.path.join(FIGURES_DIR, f"06_pca_2d_{safe_name}.png")
        plt.savefig(out2)
        plt.close()

        print(f"  ✓ Saved: {out1}")
        print(f"  ✓ Saved: {out2}")

    return pca_results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10 — Save Processed Data
# ═══════════════════════════════════════════════════════════════════════════

def section_save_processed(spectra_data: dict, pca_results: dict):
    """
    Persist all processed artefacts to ``data/processed/`` so that the
    Phase 3 model-training scripts can load them directly without having
    to re-run the full EDA pipeline.
    """
    print("\n" + "=" * 70)
    print("  SECTION 10 :  SAVING PROCESSED DATA")
    print("=" * 70)

    for (species, antibiotic), data in spectra_data.items():
        safe_sp = species.replace(" ", "_").lower()
        safe_ab = antibiotic.lower().replace(" ", "_")
        prefix  = f"{safe_sp}_{safe_ab}"

        pca_data = pca_results[(species, antibiotic)]

        # PCA-transformed feature matrix
        np.save(
            os.path.join(PROCESSED_DIR, f"{prefix}_X_processed.npy"),
            pca_data["X_pca"],
        )

        # TIC-normalised spectra (kept for optional re-processing)
        np.save(
            os.path.join(PROCESSED_DIR, f"{prefix}_X_tic.npy"),
            data["X_tic"],
        )

        # Binary labels
        np.save(
            os.path.join(PROCESSED_DIR, f"{prefix}_y.npy"),
            data["y"],
        )

        # Metadata CSV
        data["metadata"].to_csv(
            os.path.join(PROCESSED_DIR, f"{prefix}_metadata.csv"),
            index=False,
        )

        # Fitted sklearn pipeline (for inference in the API later)
        joblib.dump(
            pca_data["pipeline"],
            os.path.join(PROCESSED_DIR, f"{prefix}_pipeline.joblib"),
        )

        print(f"\n  {prefix}:")
        print(f"    ✓ X_processed.npy   shape {pca_data['X_pca'].shape}")
        print(f"    ✓ X_tic.npy         shape {data['X_tic'].shape}")
        print(f"    ✓ y.npy             {len(data['y']):,} labels")
        print(f"    ✓ metadata.csv")
        print(f"    ✓ pipeline.joblib")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN — orchestrate all EDA sections
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_plot_style()

    print("\n" + "█" * 70)
    print("  DRIAMS-A Exploratory Data Analysis")
    print("  Target: AMR Prediction from MALDI-TOF Mass Spectrometry")
    print("█" * 70)

    # 1 — Overview
    metadata = load_metadata(DRIAMS_ROOT)
    section_metadata_overview(metadata)

    # 2 — Filter targets
    target_df = section_target_filtering(metadata)

    # 3 — Class distribution
    labeled_results = section_class_distribution(target_df)

    # 4–6 — Spectra loading, TIC norm, mean spectra
    spectra_data = section_spectra_analysis(labeled_results)

    # 7 — Variance analysis
    variance_info = section_variance_analysis(spectra_data)

    # 8 — Correlation analysis
    section_correlation_analysis(spectra_data, variance_info, top_n=50)

    # 9 — PCA
    pca_results = section_pca_analysis(spectra_data)

    # 10 — Save
    section_save_processed(spectra_data, pca_results)

    print("\n" + "█" * 70)
    print("  EDA PIPELINE COMPLETE")
    print(f"  Figures saved to:        {FIGURES_DIR}")
    print(f"  Processed data saved to: {PROCESSED_DIR}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
