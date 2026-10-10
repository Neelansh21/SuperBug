"""
Machine Learning Training Pipeline for AMR Prediction
======================================================

Orchestrates model selection, cross-validation, and serialization for
species-specific antimicrobial resistance classifiers.

Candidate Classifiers:
    1. Support Vector Machine (SVM)
    2. XGBoost
    3. LightGBM
    4. CatBoost
    5. AdaBoost

Imbalance Handling:
    For highly imbalanced targets (e.g. K. pneumoniae / Meropenem at 0.9%
    resistance), SMOTE is applied inside each CV training fold to prevent
    majority class collapse while avoiding validation data leakage. Models
    also use native class weight parameters (scale_pos_weight, auto_class_weights).
"""

import os
import logging
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier

from ml_pipeline.preprocessing import build_preprocessing_pipeline


logger = logging.getLogger(__name__)



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline", "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# --- 1. Data Ingestion ---

def load_processed_data(species: str, antibiotic: str) -> tuple:
    """
    Load the PCA-processed feature matrix and binary labels for a target.

    Retrieves the preprocessed datasets and aligned clinical labels
    generated during the preprocessing phase.

    Parameters
    ----------
    species : str
        The target pathogen species name.
    antibiotic : str
        The target antibiotic name.

    Returns
    -------
    tuple
        A tuple containing (X, y, metadata) where X is the processed 
        feature matrix, y is the label array, and metadata is the 
        corresponding pandas DataFrame.
    """
    safe_sp = species.replace(" ", "_").lower()
    safe_ab = antibiotic.lower().replace(" ", "_")
    prefix = f"{safe_sp}_{safe_ab}"

    X_path = os.path.join(PROCESSED_DIR, f"{prefix}_X_processed.npy")
    y_path = os.path.join(PROCESSED_DIR, f"{prefix}_y.npy")
    meta_path = os.path.join(PROCESSED_DIR, f"{prefix}_metadata.csv")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"Processed data files not found for {species}/{antibiotic} in {PROCESSED_DIR}. "
            "Please run notebooks/eda_driams.py first to generate them."
        )

    X = np.load(X_path)
    y = np.load(y_path)
    metadata = pd.read_csv(meta_path)


    return X, y, metadata


def load_tic_data(species: str, antibiotic: str) -> tuple:
    """
    Load the TIC-normalized (pre-PCA) spectra and labels for a target.

    Cross-validation must fit the VarianceThreshold/Scaler/PCA pipeline
    *inside each fold* to avoid leaking validation-fold statistics into the
    feature transform. That requires the un-reduced TIC features rather than
    the globally-PCA'd ``X_processed`` matrix.

    Parameters
    ----------
    species : str
        The target pathogen species name.
    antibiotic : str
        The target antibiotic name.

    Returns
    -------
    tuple
        A tuple (X_tic, y) where X_tic is the TIC-normalized feature 
        matrix of shape (n_samples, 6000), and y is the binary label array.
    """
    safe_sp = species.replace(" ", "_").lower()
    safe_ab = antibiotic.lower().replace(" ", "_")
    prefix = f"{safe_sp}_{safe_ab}"

    X_tic_path = os.path.join(PROCESSED_DIR, f"{prefix}_X_tic.npy")
    y_path = os.path.join(PROCESSED_DIR, f"{prefix}_y.npy")

    if not (os.path.exists(X_tic_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"TIC data files not found for {species}/{antibiotic} in {PROCESSED_DIR}. "
            "Please run notebooks/eda_driams.py first to generate them."
        )

    X_tic = np.load(X_tic_path)
    y = np.load(y_path)

    return X_tic, y


# --- 2. Classifier Candidates Initialization ---

def get_model_candidates(scale_pos_weight: float = 1.0, clinical_class_weight: dict = None) -> dict:
    """
    Initialize the candidate model dictionary.

    Uses a mix of kernelized classifiers (SVM) and tree-based ensemble
    algorithms (XGBoost, LightGBM, CatBoost, AdaBoost). Each model is
    configured with a class-balancing scheme to penalize minority-class
    errors appropriately.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative to positive samples for tree-based models.
    clinical_class_weight : dict, optional
        Custom class weight dict (e.g., {0: 1, 1: 500}) for aggressive
        minority-class penalization. When provided, overrides "balanced"
        for SVM and derives scale_pos_weight for tree models.

    Returns
    -------
    dict
        A dictionary mapping model names (str) to instantiated scikit-learn
        compatible classifier objects.
    """
    # SVM class weight: use clinical override if provided, else balanced
    svm_class_weight = clinical_class_weight if clinical_class_weight else "balanced"

    # For tree-based models: if clinical weights provided, derive the
    # scale_pos_weight from the weight ratio (e.g., 500/1 = 500.0)
    if clinical_class_weight:
        tree_scale = clinical_class_weight.get(1, 1) / max(clinical_class_weight.get(0, 1), 1)
    else:
        tree_scale = scale_pos_weight

    # RBF kernel for non-linear decision boundaries; probability=True for
    # clinical soft-max thresholds; cache_size/tol tuned for speed
    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        class_weight=svm_class_weight,
        cache_size=2000,
        tol=1e-2,
        random_state=42,
    )

    # scale_pos_weight balances the positive (Resistant) class penalty;
    # max_depth limited to control overfitting on tabular PCA features
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=tree_scale,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    # LightGBM: efficient for high-dimensional PCA components
    lgb_model = LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=tree_scale,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    # CatBoost: automated class weight balancing, verbose=0 for clean output
    cat_model = CatBoostClassifier(
        iterations=150,
        depth=5,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_state=42,
        verbose=0,
    )

    # AdaBoost: adaptive boosting with decision tree base learners
    ada_model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )

    return {
        "SVM": svm_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "CatBoost": cat_model,
        "AdaBoost": ada_model,
    }


# --- 3. Stratified Cross-Validation & SMOTE Evaluation ---

def train_evaluate_cv(
    X_tic: np.ndarray,
    y: np.ndarray,
    species: str,
    antibiotic: str,
    n_splits: int = 5,
    threshold: float = 0.5,
    clinical_class_weight: dict = None,
) -> dict:
    """
    Run 5-Fold Stratified Cross Validation over all candidate models.

    Takes the TIC-normalized (pre-PCA) matrix ``X_tic`` and fits the full
    VarianceThreshold → StandardScaler → PCA preprocessing *inside each fold*
    on the training split only. This prevents the scaler/PCA from seeing
    validation-fold statistics — the data leakage that would otherwise
    inflate the reported CV metrics.

    If the target class imbalance is severe (minority < 5%), SMOTE is
    applied inside each training fold (never on validation splits) to
    re-balance class distributions and enhance model sensitivity.

    Parameters
    ----------
    X_tic : np.ndarray
        TIC-normalized feature matrix, shape (n_samples, 6000).
    y : np.ndarray
        Binary labels (0=Susceptible, 1=Resistant).
    species : str
        Target pathogen species name.
    antibiotic : str
        Target antibiotic name.
    n_splits : int
        Number of cross-validation folds. Defaults to 5.
    threshold : float
        Decision threshold for classifying as Resistant. Defaults to 0.5.
    clinical_class_weight : dict, optional
        Custom class weight dict. If provided, disables SMOTE.

    Returns
    -------
    dict
        A summary dictionary containing mean cross-validation metrics
        (roc_auc, avg_prec, f1, precision, recall, fpr, fnr) for each model.
    """
    X = X_tic  # alias kept for the imbalance/count bookkeeping below


    n_s = np.sum(y == 0)
    n_r = np.sum(y == 1)
    imbalance_ratio = float(n_s / max(n_r, 1))

    # Imbalance strategy is EITHER synthetic oversampling (SMOTE) OR aggressive
    # cost-sensitive weighting — never both. Stacking them double-counts the
    # minority class: SMOTE rebalances to ~50/50, and then a 500:1 weight on
    # top of balanced data pushes the boundary arbitrarily. When a caller
    # supplies clinical_class_weight, we defer entirely to cost-sensitive
    # learning and skip SMOTE.
    pct_resistant = (n_r / len(y)) * 100
    apply_smote = (pct_resistant < 5.0) and (clinical_class_weight is None)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_metrics = {
        name: {
            "roc_auc": [],
            "avg_prec": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "fpr": [],
            "fnr": [],
        }
        for name in ["SVM", "XGBoost", "LightGBM", "CatBoost", "AdaBoost"]
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        
        # STEP 1: Split the raw TIC data into training and validation sets for this fold.

        X_train_tic, X_val_tic = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # STEP 2: Fit preprocessing on the TRAINING fold only (no leakage).
        # We must build a fresh pipeline (VarianceThreshold -> Scaler -> PCA) for every fold.
        # It is fit() ONLY on X_train_tic. It is then used to transform() X_val_tic.
        # This prevents validation data statistics from leaking into the PCA components.

        fold_pipeline = build_preprocessing_pipeline()
        X_train = fold_pipeline.fit_transform(X_train_tic)
        X_val = fold_pipeline.transform(X_val_tic)

        # STEP 3: Handle extreme class imbalance using SMOTE (Synthetic Minority Over-sampling).

        if apply_smote:
            # We cap k_neighbors based on how many minority samples actually exist in this fold.
            # If a fold only has 3 resistant samples, k_neighbors must be strictly < 3.
            k_neigh = min(int(np.sum(y_train == 1)) - 1, 5)
            if k_neigh >= 1:
                sm = SMOTE(k_neighbors=k_neigh, random_state=42)
                # Resample ONLY the training data. Never synthesize validation data!
                X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            else:
                X_train_res, y_train_res = X_train, y_train
        else:
            X_train_res, y_train_res = X_train, y_train

        # STEP 4: Calculate the cost-sensitive weights for tree models based on the (potentially SMOTE'd) data.
        
        train_ns = np.sum(y_train_res == 0)
        train_nr = np.sum(y_train_res == 1)
        fold_scale_weight = float(train_ns / max(train_nr, 1))

        # Instantiate fresh model objects for this specific fold
        models = get_model_candidates(scale_pos_weight=fold_scale_weight, clinical_class_weight=clinical_class_weight)

        for name, model in models.items():
            model.fit(X_train_res, y_train_res)

            y_prob = model.predict_proba(X_val)[:, 1]
            # as it returns probabilities of both resistant and susceptible classes, we take the second column (index 1) for the positive class.

            # Use species-specific clinical threshold instead of
            # the default 0.5 baked into model.predict()

            y_pred = (y_prob >= threshold).astype(int)

            roc_auc = roc_auc_score(y_val, y_prob)
            avg_prec = average_precision_score(y_val, y_prob)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)

            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            # Major Error (FPR): unnecessary stronger treatment
            fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
            # Very Major Error (FNR): potentially fatal mistreatment
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            cv_metrics[name]["roc_auc"].append(roc_auc)
            cv_metrics[name]["avg_prec"].append(avg_prec)
            cv_metrics[name]["f1"].append(f1)
            cv_metrics[name]["precision"].append(prec)
            cv_metrics[name]["recall"].append(rec)
            cv_metrics[name]["fpr"].append(fpr)
            cv_metrics[name]["fnr"].append(fnr)

    # Compute mean metrics across folds
    cv_summary = {}
    for name in cv_metrics:
        cv_summary[name] = {
            m: np.mean(cv_metrics[name][m]) for m in cv_metrics[name]
        }

    return cv_summary


# --- 4. Final Training and Model Serialization ---

def train_and_serialize_best_model(
    X_tic: np.ndarray,
    y: np.ndarray,
    cv_results: dict,
    species: str,
    antibiotic: str,
    model_filename: str,
    clinical_class_weight: dict = None,
    data_limited: bool = False,
    data_limited_reason: str = None,
):
    """
    Identify the best candidate model, train it on ALL data, and save both the
    fitted preprocessing pipeline and the model.

    We take the TIC-normalized ``X_tic`` (not a pre-PCA'd matrix) and fit a
    fresh preprocessing pipeline (sqrt → VarianceThreshold → Scaler → PCA) on
    the full dataset here. Fitting the pipeline and the model together in one
    place guarantees the saved ``*_pipeline.joblib`` used at inference time is
    exactly the transform the model was trained on — so we never have to keep a
    separately-generated feature matrix in sync.

    The best model is the one with the highest cross-validated F1-score (best
    precision/recall trade-off). For highly imbalanced targets (prevalence < 5%
    and no explicit cost weighting), SMOTE oversamples the minority class before
    the final fit.

    Parameters
    ----------
    X_tic : np.ndarray
        TIC-normalized feature matrix.
    y : np.ndarray
        Binary labels.
    cv_results : dict
        Summary metrics from train_evaluate_cv to select the best model.
    species : str
        Target pathogen species name.
    antibiotic : str
        Target antibiotic name.
    model_filename : str
        Base filename used to save the exported joblib model.
    clinical_class_weight : dict, optional
        Custom class weight dict.
    data_limited : bool
        Flag indicating if the target lacks sufficient resistant samples.
    data_limited_reason : str, optional
        Reason for the data_limited flag, included in UI warnings.

    Returns
    -------
    None
        The function saves artifacts to disk and returns nothing.
    """
    # STEP 1: Pick the winning model based on the highest F1 score from cross-validation.
    best_model_name = max(cv_results, key=lambda k: cv_results[k]["f1"])
    best_metrics = cv_results[best_model_name]

    # STEP 2: Fit the final preprocessing pipeline on ALL available data
    # Why all data? Because this is the artifact we deploy to production.
    # The strict train/val separation was only required during evaluation above to get honest metrics.
    # For the final model, we want it to learn from every single patient sample available.

    pipeline = build_preprocessing_pipeline()
    X = pipeline.fit_transform(X_tic)

    n_s = np.sum(y == 0)
    n_r = np.sum(y == 1)

    # SMOTE and cost-sensitive weighting are mutually exclusive (see note in
    # train_evaluate_cv). If a clinical_class_weight is supplied, skip SMOTE.
    pct_resistant = (n_r / len(y)) * 100
    apply_smote = (pct_resistant < 5.0) and (clinical_class_weight is None)

    if apply_smote:
        k_neigh = min(int(n_r) - 1, 5)
        if k_neigh >= 1:
            sm = SMOTE(k_neighbors=k_neigh, random_state=42)
            X_train_final, y_train_final = sm.fit_resample(X, y)
        else:
            X_train_final, y_train_final = X, y
    else:
        X_train_final, y_train_final = X, y

    final_ns = np.sum(y_train_final == 0)
    final_nr = np.sum(y_train_final == 1)
    final_scale_weight = float(final_ns / max(final_nr, 1))

    best_model = get_model_candidates(scale_pos_weight=final_scale_weight, clinical_class_weight=clinical_class_weight)[best_model_name]


    best_model.fit(X_train_final, y_train_final)

    # Persist the fitted pipeline next to the processed data so the API loads
    # the exact transform this model was trained with.
    safe_sp = species.replace(" ", "_").lower()
    safe_ab = antibiotic.lower().replace(" ", "_")
    pipeline_path = os.path.join(PROCESSED_DIR, f"{safe_sp}_{safe_ab}_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)

    save_path = os.path.join(MODELS_DIR, f"{model_filename}_model.joblib")

    # Package model with metadata for validation reporting in the UI
    payload = {
        "model": best_model,
        "model_name": best_model_name,
        "species": species,
        "antibiotic": antibiotic,
        "metrics": best_metrics,
        "classes": [0, 1],
        "data_limited": data_limited,
        "data_limited_reason": data_limited_reason,
    }

    joblib.dump(payload, save_path)



# --- 4b. Held-out Temporal Test Evaluation ---

def evaluate_temporal_holdout(
    species: str,
    antibiotic: str,
    best_model_name: str,
    threshold: float = 0.5,
    test_year: int = 2018,
    clinical_class_weight: dict = None,
) -> dict:
    """
    Evaluate on a fully held-out *temporal* test set.

    Cross-validation reuses the same isolates for training and scoring, so it
    can still be optimistic about deployment performance. Here we train on all
    years strictly before ``test_year`` and test on ``test_year`` — a split the
    model never influenced. This also mirrors real clinical use, where the
    model must predict resistance for *future* isolates.

    Preprocessing (VarianceThreshold → Scaler → PCA) is fit on the training
    years only; SMOTE (if the target is imbalanced) is applied to the training
    partition only. The held-out year is left completely untouched.

    Parameters
    ----------
    species : str
        Target pathogen species name.
    antibiotic : str
        Target antibiotic name.
    best_model_name : str
        Name of the best performing model from CV.
    threshold : float
        Decision threshold for classifying as Resistant.
    test_year : int
        The year to hold out for testing. Defaults to 2018.
    clinical_class_weight : dict, optional
        Custom class weight dict.

    Returns
    -------
    dict
        Test-set metrics plus n_test / n_test_resistant for context, or
        ``{}`` if either partition lacks both classes.
    """
    X_tic, y = load_tic_data(species, antibiotic)
    _, _, metadata = load_processed_data(species, antibiotic)

    if "year" not in metadata.columns or len(metadata) != len(y):
        return {}

    # STEP 1: Strictly isolate data by time.
    # Train on past data (e.g. 2015-2017) and test on future data (e.g. 2018).
    # This prevents the model from "looking into the future" and proves it generalizes to new strains.
    
    years = metadata["year"].values
    train_mask = years < test_year
    test_mask = years == test_year

    y_train, y_test = y[train_mask], y[test_mask]

    # Guard: Ensure we actually have resistant samples in the test year to evaluate against.
    if len(y_test) == 0 or y_test.sum() == 0 or y_train.sum() == 0:
        return {
            "evaluable": False,
            "n_test": int(len(y_test)),
            "n_test_resistant": int(y_test.sum()),
        }

    # STEP 2: Fit preprocessing on training years ONLY. Apply transform to the test year.
    prep = build_preprocessing_pipeline()
    X_train = prep.fit_transform(X_tic[train_mask])
    X_test = prep.transform(X_tic[test_mask])

    # Rebalance the training partition only (SMOTE xor cost-sensitive weighting)
    pct_resistant = (y_train.sum() / len(y_train)) * 100
    if pct_resistant < 5.0 and clinical_class_weight is None:
        k_neigh = min(int(y_train.sum()) - 1, 5)
        if k_neigh >= 1:
            sm = SMOTE(k_neighbors=k_neigh, random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

    scale_weight = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    model = get_model_candidates(
        scale_pos_weight=scale_weight,
        clinical_class_weight=clinical_class_weight,
    )[best_model_name]
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "evaluable": True,
        "n_test": int(len(y_test)),
        "n_test_resistant": int(y_test.sum()),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_prec": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "fpr": fp / (tn + fp) if (tn + fp) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


# --- 5. Pipeline CLI Entrypoint ---

def main():
    """
    CLI Entrypoint for the training pipeline.

    Iterates over all configured targets in TARGET_CONFIG, runs CV, 
    serializes best models, evaluates temporal holdout, and prints
    a comprehensive summary report to stdout.

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

    from ml_pipeline.preprocessing import TARGET_CONFIG

    final_report = []
    holdout_report = []

    for species, config in TARGET_CONFIG.items():
        antibiotic = config["primary"]
        model_filename = config["model_filename"]
        threshold = config.get("threshold", 0.5)
        clinical_class_weight = config.get("clinical_class_weight", None)
        data_limited = config.get("data_limited", False)
        data_limited_reason = config.get("data_limited_reason", None)

        try:
            # Everything downstream works from X_tic (TIC-normalized, pre-PCA).
            # CV refits preprocessing per fold for leak-free metrics; the final
            # model + its deployed pipeline are fit on all of X_tic together.
            X_tic, y = load_tic_data(species, antibiotic)
            cv_results = train_evaluate_cv(X_tic, y, species, antibiotic, threshold=threshold, clinical_class_weight=clinical_class_weight)
            train_and_serialize_best_model(
                X_tic, y, cv_results, species, antibiotic, model_filename,
                clinical_class_weight=clinical_class_weight,
                data_limited=data_limited, data_limited_reason=data_limited_reason,
            )

            best_model_name = max(cv_results, key=lambda k: cv_results[k]["f1"])
            m = cv_results[best_model_name]
            final_report.append({
                "Species": species,
                "Antibiotic": antibiotic,
                "Best Model": best_model_name,
                "Status": "DATA-LIMITED" if data_limited else "OK",
                "ROC AUC": f"{m['roc_auc']:.3f}",
                "Avg Precision": f"{m['avg_prec']:.3f}",
                "F1-Score": f"{m['f1']:.3f}",
                "Recall (Sens.)": f"{m['recall']:.3f}",
                "Major Error (FPR)": f"{m['fpr']:.2%}",
                "V. Major Error (FNR)": f"{m['fnr']:.2%}",
            })

            # Honest generalization check: train on <2018, test on 2018
            ho = evaluate_temporal_holdout(
                species, antibiotic, best_model_name,
                threshold=threshold, test_year=2018,
                clinical_class_weight=clinical_class_weight,
            )
            if ho.get("evaluable"):
                holdout_report.append({
                    "Species": species,
                    "Antibiotic": antibiotic,
                    "Test n (2018)": ho["n_test"],
                    "Test R": ho["n_test_resistant"],
                    "ROC AUC": f"{ho['roc_auc']:.3f}",
                    "F1-Score": f"{ho['f1']:.3f}",
                    "Recall (Sens.)": f"{ho['recall']:.3f}",
                    "Major Error (FPR)": f"{ho['fpr']:.2%}",
                    "V. Major Error (FNR)": f"{ho['fnr']:.2%}",
                })
            else:
                holdout_report.append({
                    "Species": species,
                    "Antibiotic": antibiotic,
                    "Test n (2018)": ho.get("n_test", 0),
                    "Test R": ho.get("n_test_resistant", 0),
                    "ROC AUC": "n/a",
                    "F1-Score": "n/a",
                    "Recall (Sens.)": "insufficient test positives",
                    "Major Error (FPR)": "n/a",
                    "V. Major Error (FNR)": "n/a",
                })

        except Exception as exc:
            logger.exception("Training failed for %s / %s: %s", species, antibiotic, exc)

    if final_report:
        report_df = pd.DataFrame(final_report)
        print("\n" + "=" * 90)
        print("  FINAL CROSS-VALIDATION MODEL PERFORMANCE SUMMARY")
        print("=" * 90)
        print(report_df.to_string(index=False))
        print("=" * 90 + "\n")

    if holdout_report:
        holdout_df = pd.DataFrame(holdout_report)
        print("=" * 90)
        print("  HELD-OUT TEMPORAL TEST (train ≤2017 · test 2018 — never seen in training)")
        print("=" * 90)
        print(holdout_df.to_string(index=False))
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
