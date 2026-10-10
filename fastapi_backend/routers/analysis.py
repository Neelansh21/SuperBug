"""
FastAPI Analysis Router
========================

Handles all prediction-related API endpoints:

    GET  /health          — Liveness probe
    GET  /models          — List available AMR classifiers + CV metrics
    POST /predict         — JSON body with raw 6000-bin spectrum
    POST /predict/file    — Upload a .txt spectrum file from disk

Inference Pipeline (per request):
    1. Parse input spectrum (6,000 floats) from JSON or uploaded file
    2. Look up the species' preprocessing pipeline + trained model (.joblib)
    3. TIC-normalize → transform through fitted Pipeline (VarThreshold → Scaler → PCA)
    4. Run model.predict_proba() to obtain resistance probability
    5. Generate a clinical interpretation report via LangChain RAG
    6. Package result into PredictionResponse and return
"""

import os
import io
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from fastapi_backend.schemas.models import (
    PredictionRequest,
    PredictionResponse,
    ModelSummary,
    ModelsListResponse,
    HealthResponse,
    CVMetrics,
    PatientMetadata,
)
from ml_pipeline.preprocessing import TARGET_CONFIG, tic_normalize, N_BINS


logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths can be overridden via environment variables — essential for Render/cloud
# deployment where model files may live on a mounted disk or different path.
PROCESSED_DIR = os.environ.get(
    "PROCESSED_DATA_DIR",
    os.path.join(PROJECT_ROOT, "data", "processed"),
)
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.join(PROJECT_ROOT, "ml_pipeline", "models"),
)

# Resolve relative paths against PROJECT_ROOT
if not os.path.isabs(PROCESSED_DIR):
    PROCESSED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, PROCESSED_DIR))
if not os.path.isabs(MODELS_DIR):
    MODELS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, MODELS_DIR))

# In-memory caches (populated at startup via load_all_models)
_model_cache: dict = {}       # species → {"model_pkg": ..., "pipeline": ...}


def _get_species_prefix(species: str) -> str:
    """Derive the file prefix from species + primary antibiotic config."""
    cfg = TARGET_CONFIG[species]
    safe_sp = species.replace(" ", "_").lower()
    safe_ab = cfg["primary"].lower().replace(" ", "_")
    return f"{safe_sp}_{safe_ab}"


def load_all_models():
    """
    Pre-load all trained models and preprocessing pipelines into memory.

    Eager loading at startup ensures inference requests execute with
    sub-second latency instead of blocking on disk I/O.
    """
    global _model_cache
    _model_cache = {}

    for species, cfg in TARGET_CONFIG.items():
        prefix = _get_species_prefix(species)
        model_path = os.path.join(MODELS_DIR, f"{cfg['model_filename']}_model.joblib")
        pipeline_path = os.path.join(PROCESSED_DIR, f"{prefix}_pipeline.joblib")

        if not os.path.exists(model_path):
            logger.warning("Model file missing for %s, skipping: %s", species, model_path)
            continue
        if not os.path.exists(pipeline_path):
            logger.warning("Pipeline file missing for %s, skipping: %s", species, pipeline_path)
            continue

        try:
            model_pkg = joblib.load(model_path)
            pipeline = joblib.load(pipeline_path)

            _model_cache[species] = {
                "model_pkg": model_pkg,
                "pipeline": pipeline,
                "config": cfg,
            }
            logger.info("Loaded model + pipeline for %s", species)

        except Exception as exc:
            logger.exception("Failed to load model/pipeline for %s: %s", species, exc)




def _run_inference(
    species: str,
    raw_spectrum: np.ndarray,
    workstation: str,
    acquisition_date: str,
    year: int,
) -> dict:
    """
    Execute the full inference pipeline on a single raw spectrum array.

    Steps:
    1. TIC-normalise → pipeline.transform() (VarianceThreshold → Scaler → PCA)
    2. model.predict_proba()
    3. LangChain RAG report generation
    """
    if species not in _model_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No model loaded for species '{species}'. Available: {list(_model_cache.keys())}",
        )

    cache = _model_cache[species]
    model_pkg = cache["model_pkg"]
    pipeline = cache["pipeline"]
    cfg = cache["config"]

    model = model_pkg["model"]
    model_name = model_pkg.get("model_name", "Unknown")
    metrics = model_pkg.get("metrics", {})
    data_limited = cfg.get("data_limited", False)
    data_limited_reason = cfg.get("data_limited_reason")

    # 1. Reshape + TIC normalise
    X_raw = raw_spectrum.reshape(1, -1)
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    X_tic = tic_normalize(X_raw)

    # 2. Transform through fitted pipeline (VarianceThreshold → Scaler → PCA)
    X_processed = pipeline.transform(X_tic)

    # 3. Predict using species-specific clinical threshold
    proba = model.predict_proba(X_processed)[0]
    resistance_prob = float(proba[1])
    threshold = cfg.get("threshold", 0.5)
    label = int(resistance_prob >= threshold)
    pred_label = "Resistant" if label == 1 else "Susceptible"

    # 4. Generate clinical report via LangChain
    try:
        from langchain_engine.chains import generate_clinical_report

        clinical_report = generate_clinical_report(
            species=species,
            antibiotic=cfg["primary"],
            prediction_prob=resistance_prob,
            label=label,
            metadata={
                "workstation": workstation,
                "acquisition_date": acquisition_date,
                "year": year,
            },
        )
    except Exception as exc:
        clinical_report = (
            f"# Clinical Report (Fallback)\n\n"
            f"Report generation encountered an error. "
            f"Prediction: **{pred_label}** with probability **{resistance_prob:.2%}**.\n\n"
            f"Error: {exc}"
        )

    # Prepend an unmissable warning for data-limited (non-validated) targets
    if data_limited:
        warning = (
            "> [!WARNING]\n"
            "> **NOT A VALIDATED MODEL — DEMONSTRATION ONLY.**\n"
            f"> {data_limited_reason or 'Trained on too few resistant isolates to be reliable.'}\n"
            "> Do not use this prediction for clinical decisions.\n\n"
        )
        clinical_report = warning + clinical_report

    return {
        "species": species,
        "antibiotic": cfg["primary"],
        "prediction_label": pred_label,
        "probability": resistance_prob,
        "clinical_report": clinical_report,
        "model_name": model_name,
        "cv_metrics": CVMetrics(
            roc_auc=metrics.get("roc_auc", 0.0),
            avg_prec=metrics.get("avg_prec", 0.0),
            f1=metrics.get("f1", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            fpr=metrics.get("fpr", 0.0),
            fnr=metrics.get("fnr", 0.0),
            threshold=threshold,
        ),
        "patient_metadata": PatientMetadata(
            workstation=workstation,
            acquisition_date=acquisition_date,
            year=year,
        ),
        "data_limited": data_limited,
        "data_limited_reason": data_limited_reason,
    }


# --- Routes ---

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — returns current API status and model count."""
    return HealthResponse(
        status="ok",
        message="AMR Prediction API is running.",
        models_loaded=len(_model_cache),
    )


@router.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """
    List all loaded AMR classifiers with their cross-validation statistics.

    Used by the Streamlit front-end to dynamically populate the model
    selection sidebar and display validation performance badges.
    """
    summaries = []
    for species, cache in _model_cache.items():
        cfg = cache["config"]
        pkg = cache["model_pkg"]
        metrics = pkg.get("metrics", {})

        summaries.append(
            ModelSummary(
                species=species,
                antibiotic=cfg["primary"],
                clinical_name=cfg["clinical_name"],
                model_name=pkg.get("model_name", "Unknown"),
                cv_metrics=CVMetrics(
                    roc_auc=metrics.get("roc_auc", 0.0),
                    avg_prec=metrics.get("avg_prec", 0.0),
                    f1=metrics.get("f1", 0.0),
                    precision=metrics.get("precision", 0.0),
                    recall=metrics.get("recall", 0.0),
                    fpr=metrics.get("fpr", 0.0),
                    fnr=metrics.get("fnr", 0.0),
                    threshold=cfg.get("threshold", 0.5),
                ),
                model_filename=cfg["model_filename"],
                data_limited=cfg.get("data_limited", False),
                data_limited_reason=cfg.get("data_limited_reason"),
            )
        )

    return ModelsListResponse(models=summaries, count=len(summaries))


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_json(request: PredictionRequest):
    """
    Accept a JSON payload with a raw 6,000-bin MALDI-TOF spectrum and return
    the AMR prediction, probability, and clinical interpretation report.
    """
    raw_spectrum = np.array(request.spectrum, dtype=np.float64)

    result = _run_inference(
        species=request.species,
        raw_spectrum=raw_spectrum,
        workstation=request.workstation,
        acquisition_date=request.acquisition_date,
        year=request.year,
    )

    return PredictionResponse(**result)


@router.post("/predict/file", response_model=PredictionResponse, tags=["Prediction"])
async def predict_file(
    file: UploadFile = File(..., description="Tab/space-separated .txt spectrum file."),
    species: str = Form(..., description="Target pathogen species name."),
    workstation: str = Form(default="General Ward", description="Hospital ward."),
    acquisition_date: str = Form(default="2026-05-28", description="Acquisition date."),
    year: int = Form(default=2018, description="Acquisition year."),
):
    """
    Accept a DRIAMS-format .txt spectrum file upload and return the AMR
    prediction with a clinical report.

    Parses the ``binned_intensity`` column from the uploaded text file
    (matching the DRIAMS-A binned_6000 format).
    """
    if species not in TARGET_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown species '{species}'. Supported: {list(TARGET_CONFIG.keys())}",
        )

    try:
        content = await file.read()
        text = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=0)

        if "binned_intensity" not in df.columns:
            raise ValueError(
                f"Column 'binned_intensity' not found. "
                f"Available columns: {list(df.columns)}"
            )

        raw_spectrum = df["binned_intensity"].values.astype(np.float64)

        if len(raw_spectrum) != N_BINS:
            raise ValueError(
                f"Expected {N_BINS} intensity values, got {len(raw_spectrum)}."
            )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse uploaded file: {exc}",
        )

    result = _run_inference(
        species=species,
        raw_spectrum=raw_spectrum,
        workstation=workstation,
        acquisition_date=acquisition_date,
        year=year,
    )

    return PredictionResponse(**result)
