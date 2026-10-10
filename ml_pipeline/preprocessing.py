"""
MALDI-TOF Mass Spectrometry Preprocessing Pipeline
====================================================

Reusable data-processing backbone for the AMR prediction project.
Every function is purpose-built for the DRIAMS-A dataset structure
(binned_6000 spectra + id/ metadata CSVs).

Pipeline stages:
    1. Metadata loading  — concatenate *_strat.csv across years
    2. Species filtering  — strict equality match on target pathogens
    3. Label binarization — S→0, R→1, drop I / - / empty / mixed
    4. Spectrum ingestion  — read binned_6000/*.txt into NumPy matrices
    5. TIC normalization   — divide each spectrum by its total ion current
    6. Sklearn pipeline    — VarianceThreshold → StandardScaler → PCA (95%)
"""

import os
import re
import logging


import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


# --- Constants & Target Configuration ---

YEARS = [2015, 2016, 2017, 2018]
N_BINS = 6000  # number of m/z bins per spectrum

# Each species maps to its primary antibiotic column in the DRIAMS metadata
# and a clinical name used in LangChain prompt generation.
TARGET_CONFIG = {
    "Staphylococcus aureus": {
        "antibiotics": ["Cefoxitin_screen", "Oxacillin"],
        "primary": "Oxacillin",
        "clinical_name": "MRSA Detection",
        "model_filename": "s_aureus_oxacillin",
        # Optimized via leak-free OOF cross-validation (threshold.py).
        # Best F1 with FPR ≤ 10%: T=0.50 → F1=0.725, Recall=69.6%, FPR=5.3%
        "threshold": 0.50,
    },
    "Klebsiella pneumoniae": {
        "antibiotics": ["Meropenem"],
        "primary": "Meropenem",
        "clinical_name": "Carbapenem Resistance",
        "model_filename": "k_pneumoniae_meropenem",
        # Lowered well below 0.5 for clinical sensitivity. With <1% resistance
        # prevalence the model rarely pushes probabilities high, so a low
        # threshold flags any sample with modest resistance suspicion for
        # confirmatory culture, reducing the False Negative Rate (VME).
        # Leak-free OOF best F1 (FPR ≤ 10%): T=0.15. NOTE: still data-limited —
        # see data_limited below; treat this threshold as illustrative only.
        "threshold": 0.15,
        # Custom 500:1 clinical cost weight for extreme imbalance (0.85% R).
        # 'balanced' gives ~116:1 which is insufficient for SVM Platt scaling
        # to push resistant probabilities above the decision threshold.
        # NOTE: cost-sensitive weighting is used INSTEAD OF SMOTE here, not in
        # addition to it (stacking both double-counts the minority class).
        "clinical_class_weight": {0: 1, 1: 500},
        # Only 24 resistant isolates in the entire dataset (~0.85%). Any CV /
        # holdout metric derived from ~5 positives per fold is statistically
        # meaningless. This flag marks the target as NOT a validated model —
        # reporting and the UI should present it as a data limitation, not a
        # deployable classifier. Retained only to demonstrate the pipeline on
        # an extreme-imbalance case.
        "data_limited": True,
        "data_limited_reason": "Only 24 resistant isolates (~0.85%); metrics are not statistically reliable.",
    },
    "Escherichia coli": {
        "antibiotics": ["Ceftriaxone", "Ciprofloxacin"],
        "primary": "Ceftriaxone",
        "clinical_name": "3GC / Fluoroquinolone Resistance",
        "model_filename": "e_coli_ceftriaxone",
        # Optimized via leak-free OOF cross-validation (threshold.py).
        # Best F1 with FPR ≤ 10%: T=0.45 → F1=0.715, Recall=66.7%, FPR=5.5%
        "threshold": 0.45,
    },
}

TARGET_SPECIES = list(TARGET_CONFIG.keys())

# Regex to detect replicate-discordant labels like "R(1), S(1)" where the lab
# ran the same test twice with conflicting results — clinically ambiguous, must drop.
_MIXED_LABEL_RE = re.compile(r"[RSI]\(\d+\)")


# --- 1. Metadata Loading ---

def load_metadata(
    driams_root: str,
    years: list = None,
    use_strat: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate DRIAMS-A metadata CSV files across years.

    Uses *_strat.csv by default because it includes the ``workstation``
    column (Blood / Urine / Respiratory / …), which provides valuable
    clinical context for LangChain prompt generation.

    Parameters
    ----------
    driams_root : str
        Path to the DRIAMS-A root directory (e.g. ".../DRIAMS-A").
    years : list[int], optional
        Years to load.  Defaults to [2015, 2016, 2017, 2018].
    use_strat : bool
        True  → load ``{year}_strat.csv`` (includes workstation metadata).
        False → load ``{year}_clean.csv``.

    Returns
    -------
    pd.DataFrame
        Concatenated metadata with an added ``year`` column.
    """
    if years is None:
        years = YEARS

    suffix = "_strat.csv" if use_strat else "_clean.csv"
    frames = []

    for year in years:
        filepath = os.path.join(driams_root, "id", str(year), f"{year}{suffix}")
        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath, low_memory=False)
        df["year"] = year
        frames.append(df)


    if not frames:
        raise FileNotFoundError(
            f"No metadata files found under {driams_root}/id/"
        )

    combined = pd.concat(frames, ignore_index=True)

    return combined


# --- 2. Species Filtering ---

def filter_target_species(
    metadata: pd.DataFrame,
    species_list: list = None,
) -> pd.DataFrame:
    """
    Keep only rows whose ``species`` column exactly matches a target pathogen.

    Uses strict equality match rather than substring search because the DRIAMS
    species column sometimes contains prefixed entries like ``MIX!Klebsiella
    pneumoniae`` (mixed cultures) that must be excluded from clean training data.

    Parameters
    ----------
    metadata : pd.DataFrame
    species_list : list[str], optional
        Defaults to the three target species.

    Returns
    -------
    pd.DataFrame
    """
    if species_list is None:
        species_list = TARGET_SPECIES

    mask = metadata["species"].isin(species_list)
    filtered = metadata[mask].copy()



    return filtered


# --- 3. Label Binarization ---

def _binarize_single(value) -> float:
    """Map one raw antibiotic-susceptibility cell to 0 / 1 / NaN."""
    if pd.isna(value):
        return np.nan

    val = str(value).strip()

    if val == "S":
        return 0.0
    if val == "R":
        return 1.0
    if val in ("I", "-", ""):
        return np.nan
    # Replicate discordance, e.g. "R(1), S(1)" — ambiguous, drop
    if _MIXED_LABEL_RE.search(val):
        return np.nan
    return np.nan


def binarize_labels(
    metadata: pd.DataFrame,
    antibiotic_col: str,
) -> pd.DataFrame:
    """
    Map antibiotic susceptibility labels to binary and drop invalid rows.

    Drops Intermediate (I) labels rather than merging them with Resistant
    because EUCAST defines I as "susceptible at increased exposure," which
    is pharmacologically distinct from true resistance. Merging would
    contaminate the Resistant class with treatable samples.

    Label mapping:
        S     → 0   Susceptible
        R     → 1   Resistant
        I     → —   Dropped (Intermediate / dose-dependent)
        -     → —   Dropped (Not tested)
        empty → —   Dropped (No AST performed)
        mixed → —   Dropped (Replicate discordance)

    Parameters
    ----------
    metadata : pd.DataFrame
    antibiotic_col : str
        Column name for the target antibiotic.

    Returns
    -------
    pd.DataFrame
        Only rows with a valid binary ``label`` column (0 or 1).
    """
    if antibiotic_col not in metadata.columns:
        available = [
            c for c in metadata.columns
            if not c.startswith("Unnamed")
        ]
        raise ValueError(
            f"Antibiotic column '{antibiotic_col}' not found in metadata.\n"
            f"Available columns: {available}"
        )

    df = metadata.copy()
    df["label"] = df[antibiotic_col].apply(_binarize_single)

    n_before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    n_dropped = n_before - len(df)

    n_s = (df["label"] == 0).sum()
    n_r = (df["label"] == 1).sum()



    return df


# --- 4. Spectrum Loading ---

def load_single_spectrum(filepath: str) -> np.ndarray:
    """
    Read one binned MALDI-TOF spectrum .txt file.

    Uses ``sep=r"\\s+"`` because DRIAMS files use a mix of tabs and spaces
    depending on the year of collection.

    Returns
    -------
    np.ndarray
        Array of 6,000 intensity values.
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=0)
    return df["binned_intensity"].values


def load_spectra_batch(
    metadata: pd.DataFrame,
    driams_root: str,
) -> tuple:
    """
    Load spectra for every sample in *metadata*.

    File path mapping: ``DRIAMS-A/binned_6000/{year}/{code}.txt``

    Parameters
    ----------
    metadata : pd.DataFrame
        Must contain ``code`` and ``year`` columns.
    driams_root : str
        Path to DRIAMS-A root.

    Returns
    -------
    X : np.ndarray, shape (n_loaded, 6000)
    valid_idx : list[int]
        Original DataFrame indices that were successfully loaded.
    """
    spectra = []
    valid_idx = []
    missing = 0

    for idx, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Loading spectra",
        unit="file",
    ):
        code = str(row["code"])
        year = int(row["year"])
        filepath = os.path.join(
            driams_root, "binned_6000", str(year), f"{code}.txt"
        )

        if not os.path.exists(filepath):
            missing += 1
            continue

        try:
            intensities = load_single_spectrum(filepath)
            if len(intensities) == N_BINS:
                spectra.append(intensities)
                valid_idx.append(idx)
            else:
                logger.warning(
                    "Skipping %s: expected %d bins, got %d",
                    filepath, N_BINS, len(intensities),
                )
        except Exception as exc:
            logger.warning("Failed to load spectrum %s: %s", filepath, exc)

    X = (
        np.array(spectra, dtype=np.float64)
        if spectra
        else np.empty((0, N_BINS))
    )

    return X, valid_idx


# --- 5. TIC Normalization ---

def tic_normalize(X: np.ndarray) -> np.ndarray:
    """
    Total Ion Current normalization.

    MALDI-TOF instruments produce spectra with varying total ion counts
    between runs and samples. TIC removes this batch effect while
    preserving relative peak ratios.

    Formula: x_norm_i = xᵢ / Σⱼ xⱼ   (each spectrum sums to 1 after)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray
        TIC-normalized matrix (same shape).
    """
    row_sums = X.sum(axis=1, keepdims=True)
    # Guard against division by zero for all-zero spectra (rare but possible
    # if a MALDI acquisition failed completely)
    row_sums[row_sums == 0] = 1.0
    return X / row_sums


# --- 6. Sklearn Preprocessing Pipeline ---

def build_preprocessing_pipeline(
    variance_threshold: float = 1e-10,
    n_components: float = 0.95,
) -> Pipeline:
    """
    Build the 4-stage scikit-learn preprocessing pipeline.

    Stage 0 — Square-root variance stabilization (NEW):
        MALDI-TOF peak intensities are extremely right-skewed — a handful of
        tall peaks are orders of magnitude larger than the rest. Feeding that
        raw skew into StandardScaler + PCA lets those few peaks dominate the
        principal components. Taking the square root of each (TIC-normalized)
        intensity compresses the tall peaks and lifts the small ones, so more
        peaks contribute useful signal. sqrt is the standard variance-
        stabilizing transform for normalized/compositional spectra (it is the
        transform behind the Hellinger distance), and on our data it nudged
        E. coli ROC-AUC up by ~0.015 while leaving the other targets unchanged.

    Stage 1 — VarianceThreshold:
        MALDI-TOF spectra contain many near-zero-variance bins in the baseline
        noise region. Eliminating them before PCA speeds up computation.

    Stage 2 — StandardScaler:
        Required before PCA so high-intensity m/z bins don't dominate
        principal components purely due to their scale.

    Stage 3 — PCA at 95% explained variance:
        Compresses the feature space from thousands of bins down to
        roughly 50–100 principal components.

    Parameters
    ----------
    variance_threshold : float
        Minimum variance to retain a feature.
    n_components : float or int
        If float in (0, 1) → fraction of variance to retain.
        If int             → exact number of components.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        # np.sqrt is applied element-wise; inputs are TIC-normalized (>= 0),
        # so the square root is always well-defined.
        ("variance_stabilizer", FunctionTransformer(np.sqrt)),
        ("variance_filter", VarianceThreshold(threshold=variance_threshold)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])


# --- 7. Full Preprocessing Orchestrator ---

def preprocess_for_species(
    driams_root: str,
    species: str,
    antibiotic: str,
    years: list = None,
    variance_threshold: float = 1e-10,
    n_pca_components: float = 0.95,
) -> dict:
    """
    End-to-end preprocessing for one species–antibiotic pair.

    Steps:
        1. Load metadata across all years
        2. Filter for the specified species
        3. Binarize labels for the specified antibiotic
        4. Load corresponding binned spectra (.txt → NumPy)
        5. Safely-impute NaNs with 0
        6. TIC normalize
        7. Fit & transform: VarianceThreshold → StandardScaler → PCA

    Parameters
    ----------
    driams_root : str
    species : str
    antibiotic : str
    years : list[int], optional
    variance_threshold : float
    n_pca_components : float or int

    Returns
    -------
    dict
        X_raw, X_tic, X_processed, y, metadata, pipeline
    """


    # 1 — metadata
    metadata = load_metadata(driams_root, years=years)

    # 2 — species filter
    species_df = filter_target_species(metadata, [species])

    # 3 — label binarization
    labeled_df = binarize_labels(species_df, antibiotic)
    if len(labeled_df) == 0:
        raise ValueError(
            f"No valid labeled samples for {species} / {antibiotic}"
        )

    # 4 — spectrum loading
    X_raw, valid_idx = load_spectra_batch(labeled_df, driams_root)
    labeled_df = labeled_df.loc[valid_idx].reset_index(drop=True)
    y = labeled_df["label"].values

    # 5 — safety imputation (edge-case guard)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # 6 — TIC normalization
    X_tic = tic_normalize(X_raw)

    # 7 — sklearn pipeline
    pipeline = build_preprocessing_pipeline(
        variance_threshold=variance_threshold,
        n_components=n_pca_components,
    )
    X_processed = pipeline.fit_transform(X_tic)

    vf = pipeline.named_steps["variance_filter"]
    pca = pipeline.named_steps["pca"]



    return {
        "X_raw": X_raw,
        "X_tic": X_tic,
        "X_processed": X_processed,
        "y": y,
        "metadata": labeled_df,
        "pipeline": pipeline,
    }
