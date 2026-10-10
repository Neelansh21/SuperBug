"""
Training Unit Tests
====================

Verifies the integrity of model training, cross-validation evaluation,
and package serialization routines.
"""

import os
import tempfile
import joblib
import numpy as np
import pytest

from ml_pipeline.training import (
    get_model_candidates,
    train_evaluate_cv,
    train_and_serialize_best_model,
)


def test_get_model_candidates():
    """Verify that all five candidate classifiers are instantiated."""
    models = get_model_candidates(scale_pos_weight=2.5)

    assert len(models) == 5
    assert "SVM" in models
    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "CatBoost" in models
    assert "AdaBoost" in models

    # Check that scale_pos_weight is propagated to boosters
    assert models["XGBoost"].scale_pos_weight == 2.5
    assert models["LightGBM"].scale_pos_weight == 2.5


def test_train_evaluate_cv():
    """Verify cross-validation execution and performance metric calculations."""
    np.random.seed(42)

    # Synthetic dataset: 100 samples, 10 features, 10% minority ratio.
    # Values are non-negative to mimic real TIC-normalized spectra (the
    # preprocessing pipeline's sqrt stage requires non-negative input).
    X = np.random.rand(100, 10)
    y = np.array([0] * 90 + [1] * 10)

    cv_results = train_evaluate_cv(X, y, "Synthetic Pathogen", "TestAntibiotic", n_splits=3)

    assert len(cv_results) == 5
    for name in ["SVM", "XGBoost", "LightGBM", "CatBoost", "AdaBoost"]:
        assert name in cv_results
        metrics = cv_results[name]

        # All mandatory metrics should be present
        assert "roc_auc" in metrics
        assert "avg_prec" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "fpr" in metrics
        assert "fnr" in metrics

        # All scores should fall within [0, 1]
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["fpr"] <= 1.0
        assert 0.0 <= metrics["fnr"] <= 1.0


def test_train_and_serialize_best_model():
    """Verify selection, retraining, and disk serialization of the best model."""
    np.random.seed(42)
    # Non-negative, mimicking TIC-normalized spectra (sqrt stage needs x >= 0)
    X = np.random.rand(80, 5)
    y = np.array([0] * 70 + [1] * 10)

    # Dummy CV scores to force selection of LightGBM (highest F1)
    dummy_cv = {
        "SVM": {"f1": 0.1, "roc_auc": 0.5},
        "XGBoost": {"f1": 0.2, "roc_auc": 0.6},
        "LightGBM": {"f1": 0.9, "roc_auc": 0.95},
        "CatBoost": {"f1": 0.3, "roc_auc": 0.7},
        "AdaBoost": {"f1": 0.15, "roc_auc": 0.55},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        import ml_pipeline.training
        original_models_dir = ml_pipeline.training.MODELS_DIR
        original_processed_dir = ml_pipeline.training.PROCESSED_DIR
        # The function now saves BOTH the model and its fitted preprocessing
        # pipeline, so redirect both output dirs into the temp folder.
        ml_pipeline.training.MODELS_DIR = tmpdir
        ml_pipeline.training.PROCESSED_DIR = tmpdir

        try:
            train_and_serialize_best_model(
                X, y, dummy_cv, "TestSpecies", "TestAntibiotic", "test_filename"
            )

            save_path = os.path.join(tmpdir, "test_filename_model.joblib")
            assert os.path.exists(save_path)

            # The matching preprocessing pipeline must also be persisted
            pipeline_path = os.path.join(tmpdir, "testspecies_testantibiotic_pipeline.joblib")
            assert os.path.exists(pipeline_path)

            payload = joblib.load(save_path)
            assert payload["model_name"] == "LightGBM"
            assert payload["species"] == "TestSpecies"
            assert payload["antibiotic"] == "TestAntibiotic"
            assert "metrics" in payload

            # Verify the reloaded pipeline + model produce predictions end-to-end
            pipeline = joblib.load(pipeline_path)
            model = payload["model"]
            preds = model.predict(pipeline.transform(X))
            assert len(preds) == len(X)

        finally:
            ml_pipeline.training.MODELS_DIR = original_models_dir
            ml_pipeline.training.PROCESSED_DIR = original_processed_dir
