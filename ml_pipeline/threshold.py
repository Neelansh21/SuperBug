"""
Clinical Threshold Optimization via Out-of-Fold Cross-Validation
=================================================================

Determines the optimal clinical decision threshold for each species by
sweeping 0.10 → 0.50 (step 0.05) using ONLY held-out test fold
probabilities from cross-validation — preventing any data leakage.

Algorithm:
    1. Load preprocessed PCA features + labels for each species
    2. Read the best model name from the existing .joblib package
    3. Generate out-of-fold (OOF) probabilities via 5-Fold Stratified CV
       (SMOTE applied inside training folds only, never on validation)
    4. Sweep thresholds on the full OOF probability vector
    5. Select threshold: best F1 among those with FPR ≤ 10% (clinical constraint)
    6. Update .joblib model package with optimized metrics + threshold

Run:
    .venv/bin/python -m ml_pipeline.threshold
"""

import os
import logging

import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

from ml_pipeline.preprocessing import TARGET_CONFIG, build_preprocessing_pipeline
from ml_pipeline.training import load_tic_data, get_model_candidates

logger = logging.getLogger(__name__)



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline", "models")

# Threshold search space: 0.10 to 0.50 in steps of 0.05
THRESHOLD_RANGE = np.arange(0.10, 0.55, 0.05)

# Clinical constraint: Major Error (FPR) should not exceed 10% -> unable to optimise for now due to data limitations, but this is the ideal constraint to apply in production settings
MAX_ACCEPTABLE_FPR = 0.10


def _generate_oof_probabilities(
    X_tic: np.ndarray,
    y: np.ndarray,
    best_model_name: str,
    n_splits: int = 5,
    clinical_class_weight: dict = None,
) -> np.ndarray:
    """
    Generate out-of-fold (OOF) probabilities using the best model.

    Each sample gets a probability from a model that never saw it during
    training. The VarianceThreshold/Scaler/PCA preprocessing is *also* refit
    inside each fold on the training split only, so neither the model nor the
    feature transform leaks validation data into threshold optimization.

    Parameters
    ----------
    X_tic : np.ndarray
        TIC-normalized (pre-PCA) feature matrix, shape (n_samples, 6000).
    y : np.ndarray
        Binary labels (0=Susceptible, 1=Resistant).
    best_model_name : str
        Name of the winning classifier (e.g. "SVM", "LightGBM").
    n_splits : int
        Number of CV folds (default: 5).
    clinical_class_weight : dict, optional
        Custom class weight dict.

    Returns
    -------
    np.ndarray
        OOF probability vector with shape (n_samples,).
    """
    oof_probs = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_r = np.sum(y == 1)
    pct_resistant = (n_r / len(y)) * 100
    # SMOTE xor cost-sensitive weighting — never both (see note in training.py)
    apply_smote = (pct_resistant < 5.0) and (clinical_class_weight is None)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_tic, y)):
        X_train_tic, X_val_tic = X_tic[train_idx], X_tic[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Refit preprocessing on the training fold only (no leakage)
        fold_pipeline = build_preprocessing_pipeline()
        X_train = fold_pipeline.fit_transform(X_train_tic)
        X_val = fold_pipeline.transform(X_val_tic)

        # Apply SMOTE inside training fold only (never on validation)
        if apply_smote:
            k_neigh = min(int(np.sum(y_train == 1)) - 1, 5)
            if k_neigh >= 1:
                sm = SMOTE(k_neighbors=k_neigh, random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)

        # Compute class weight for the training fold
        n_s_train = np.sum(y_train == 0)
        n_r_train = np.sum(y_train == 1)
        scale_weight = float(n_s_train / max(n_r_train, 1))

        # Instantiate only the best model (not all 5 candidates)
        model = get_model_candidates(scale_pos_weight=scale_weight, clinical_class_weight=clinical_class_weight)[best_model_name]
        model.fit(X_train, y_train)

        # Store held-out predictions (no leakage — model never saw val data)
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    return oof_probs


def _evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Compute clinical metrics at a specific threshold on OOF predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    threshold : float
        The decision threshold to evaluate.

    Returns
    -------
    dict
        A dictionary containing keys: threshold, f1, precision, recall, fpr, fnr.
    """
    y_pred = (y_prob >= threshold).astype(int)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": threshold,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "fnr": fnr,
    }


def optimize_threshold_for_species(species: str, config: dict) -> dict:
    """
    Run the full threshold optimization pipeline for one species.

    Parameters
    ----------
    species : str
        Species name (e.g. "Staphylococcus aureus").
    config : dict
        Entry from TARGET_CONFIG for this species.

    Returns
    -------
    dict
        Optimal threshold and associated metrics.
    """
    antibiotic = config["primary"]
    model_filename = config["model_filename"]
    model_path = os.path.join(MODELS_DIR, f"{model_filename}_model.joblib")

    # Load existing model package to determine best model name
    if not os.path.exists(model_path):
        return {}

    model_pkg = joblib.load(model_path)
    best_model_name = model_pkg.get("model_name", "SVM")



    # Load TIC (pre-PCA) data so preprocessing can be refit inside each fold
    X_tic, y = load_tic_data(species, antibiotic)


    # Generate leak-free OOF probabilities
    clinical_class_weight = config.get("clinical_class_weight", None)
    oof_probs = _generate_oof_probabilities(X_tic, y, best_model_name, clinical_class_weight=clinical_class_weight)

    # Threshold-independent metrics (computed on raw OOF probabilities)
    roc_auc = roc_auc_score(y, oof_probs)
    avg_prec = average_precision_score(y, oof_probs)

    # Sweep thresholds and collect results


    sweep_results = []
    for t in THRESHOLD_RANGE:
        result = _evaluate_threshold(y, oof_probs, t)
        sweep_results.append(result)

    # Selection: best F1 among thresholds with FPR ≤ 10%
    acceptable = [r for r in sweep_results if r["fpr"] <= MAX_ACCEPTABLE_FPR]

    if acceptable:
        best = max(acceptable, key=lambda r: r["f1"])
    else:
        # Fallback: pick threshold with lowest FPR
        best = min(sweep_results, key=lambda r: r["fpr"])

    # Combine threshold-dependent and independent metrics
    optimal_metrics = {
        "roc_auc": roc_auc,
        "avg_prec": avg_prec,
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "fpr": best["fpr"],
        "fnr": best["fnr"],
        "threshold": best["threshold"],
    }



    # Update the .joblib model package with optimized metrics
    model_pkg["metrics"] = optimal_metrics
    joblib.dump(model_pkg, model_path)

    return optimal_metrics


def main():
    """
    Run threshold optimization for all configured species.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


    results_summary = []

    for species, config in TARGET_CONFIG.items():
        try:
            metrics = optimize_threshold_for_species(species, config)
            if metrics:
                results_summary.append({
                    "Species": species,
                    "Antibiotic": config["primary"],
                    "Threshold": f"{metrics['threshold']:.2f}",
                    "ROC AUC": f"{metrics['roc_auc']:.3f}",
                    "AUPRC": f"{metrics['avg_prec']:.3f}",
                    "F1": f"{metrics['f1']:.3f}",
                    "Recall": f"{metrics['recall']:.3f}",
                    "FPR": f"{metrics['fpr']:.1%}",
                    "FNR": f"{metrics['fnr']:.1%}",
                })
        except Exception as exc:
            logger.exception(
                "Threshold optimization failed for %s: %s", species, exc
            )

    if results_summary:
        import pandas as pd
        print("\n" + "=" * 90)
        print("  OPTIMAL THRESHOLD SUMMARY (Leak-Free OOF Evaluation)")
        print("=" * 90)
        print(pd.DataFrame(results_summary).to_string(index=False))
        print("=" * 90)

        # Print values to paste into TARGET_CONFIG
        print("\n  Update TARGET_CONFIG thresholds to:")
        for r in results_summary:
            print(f"    {r['Species']}: threshold = {r['Threshold']}")
        print()


if __name__ == "__main__":
    main()
