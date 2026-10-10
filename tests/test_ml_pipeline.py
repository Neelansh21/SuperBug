"""
Preprocessing Unit Tests
========================

Verifies that the core mathematical transformations and data filtering
logic in ml_pipeline/preprocessing.py function correctly.
"""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.preprocessing import (
    _binarize_single,
    tic_normalize,
    build_preprocessing_pipeline,
)


def test_binarize_single():
    """Verify raw AMR category mapping to binary values."""
    assert _binarize_single("S") == 0.0
    assert _binarize_single("R") == 1.0

    # Intermediate, missing, and discordant entries should map to NaN
    assert np.isnan(_binarize_single("I"))
    assert np.isnan(_binarize_single("-"))
    assert np.isnan(_binarize_single(""))
    assert np.isnan(_binarize_single(np.nan))
    assert np.isnan(_binarize_single("R(1), S(1)"))
    assert np.isnan(_binarize_single("S(2), R(1)"))


def test_tic_normalize():
    """Verify that Total Ion Current normalization rescales rows to sum to 1."""
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [0.0, 0.0, 0.0, 0.0],  # Edge case: all-zero spectrum
    ])

    X_norm = tic_normalize(X)

    # Non-zero rows should sum to exactly 1.0
    assert np.allclose(X_norm[0].sum(), 1.0)
    assert np.allclose(X_norm[1].sum(), 1.0)

    # All-zero row should remain all zeros (no division error)
    assert np.allclose(X_norm[2], 0.0)


def test_build_preprocessing_pipeline():
    """Verify the architecture and configuration of the scikit-learn pipeline."""
    pipeline = build_preprocessing_pipeline(variance_threshold=1e-5, n_components=2)

    steps = dict(pipeline.steps)

    # Verify that the 3 critical stages are correctly ordered
    assert "variance_filter" in steps
    assert "scaler" in steps
    assert "pca" in steps

    # Verify that threshold parameters are propagated properly
    assert steps["variance_filter"].threshold == 1e-5
    assert steps["pca"].n_components == 2
