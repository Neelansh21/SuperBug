"""
Pydantic Request / Response Schemas
====================================

Strict validation on incoming prediction requests and clean,
typed response payloads for the FastAPI backend.

Validation Rules:
- Species names validated against the canonical TARGET_CONFIG keys
- Raw spectra must contain exactly 6,000 floating-point intensity values
  (matching the binned_6000 MALDI-TOF format from DRIAMS-A)
- Patient metadata fields default to reasonable values for rapid screening
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# Valid species — kept in sync with preprocessing.TARGET_CONFIG
VALID_SPECIES = [
    "Staphylococcus aureus",
    "Klebsiella pneumoniae",
    "Escherichia coli",
]

N_SPECTRUM_BINS = 6000


# --- Request Schemas ---

class PredictionRequest(BaseModel):
    """
    JSON body for ``POST /predict``.

    workstation, acquisition_date, and year are optional with sensible
    defaults to reduce friction during rapid clinical triage.
    """

    species: str = Field(
        ...,
        description="Target pathogen species name.",
        examples=["Staphylococcus aureus"],
    )
    spectrum: list[float] = Field(
        ...,
        description="Raw MALDI-TOF mass spectrum (6,000 binned intensity values).",
        min_length=N_SPECTRUM_BINS,
        max_length=N_SPECTRUM_BINS,
    )
    workstation: str = Field(
        default="General Ward",
        description="Hospital workstation or ward of origin.",
    )
    acquisition_date: str = Field(
        default="2026-05-28",
        description="Sample acquisition date (YYYY-MM-DD).",
    )
    year: int = Field(
        default=2018,
        description="Acquisition year (used for provenance tracking).",
    )

    @field_validator("species")
    @classmethod
    def validate_species(cls, v: str) -> str:
        if v not in VALID_SPECIES:
            raise ValueError(
                f"Unknown species '{v}'. "
                f"Supported: {VALID_SPECIES}"
            )
        return v


# --- Response Schemas ---

class CVMetrics(BaseModel):
    """Cross-validation metrics snapshot for the model that produced the prediction."""

    roc_auc: float = Field(..., description="Mean ROC AUC across CV folds.")
    avg_prec: float = Field(..., description="Mean Average Precision across CV folds.")
    f1: float = Field(..., description="Mean F1-Score across CV folds.")
    precision: float = Field(..., description="Mean Precision across CV folds.")
    recall: float = Field(..., description="Mean Recall (Sensitivity) across CV folds.")
    fpr: float = Field(..., description="Mean False Positive Rate (Major Error).")
    fnr: float = Field(..., description="Mean False Negative Rate (Very Major Error).")
    threshold: float = Field(default=0.5, description="Optimal clinical decision threshold.")


class PatientMetadata(BaseModel):
    """Patient / sample metadata echoed back in responses."""

    workstation: str
    acquisition_date: str
    year: int


class PredictionResponse(BaseModel):
    """
    JSON response from ``POST /predict`` and ``POST /predict/file``.

    Contains the raw prediction, model validation metrics, and a
    Markdown-formatted clinical report for rendering in the UI.
    """

    species: str = Field(..., description="Pathogen species name.")
    antibiotic: str = Field(..., description="Target antibiotic tested.")
    prediction_label: str = Field(
        ...,
        description="Human-readable prediction: 'Resistant' or 'Susceptible'.",
    )
    probability: float = Field(
        ...,
        description="Model-predicted probability of resistance (0.0–1.0).",
    )
    clinical_report: str = Field(
        ...,
        description="Markdown-formatted clinical interpretation report (RAG-generated).",
    )
    model_name: str = Field(
        ...,
        description="Name of the ML classifier used (e.g. 'LightGBM').",
    )
    cv_metrics: CVMetrics = Field(
        ...,
        description="Cross-validation performance metrics of the model.",
    )
    patient_metadata: PatientMetadata = Field(
        ...,
        description="Echoed patient/sample metadata.",
    )
    data_limited: bool = Field(
        default=False,
        description="True when the model was trained on too few resistant isolates "
                    "to be statistically reliable — treat the prediction as a demo, "
                    "not a validated result.",
    )
    data_limited_reason: Optional[str] = Field(
        default=None,
        description="Explanation of the data limitation, if any.",
    )


# --- Model Info Schema (for GET /models) ---

class ModelSummary(BaseModel):
    """
    Summary card for a single serialized AMR classifier.
    Returned by ``GET /models`` for the front-end to display available
    diagnostics and their validation statistics.
    """

    species: str
    antibiotic: str
    clinical_name: str
    model_name: str
    cv_metrics: Optional[CVMetrics] = None
    model_filename: str
    data_limited: bool = False
    data_limited_reason: Optional[str] = None


class ModelsListResponse(BaseModel):
    """Wrapper response for ``GET /models``."""

    models: list[ModelSummary]
    count: int


class HealthResponse(BaseModel):
    """Response from ``GET /health``."""

    status: str = "ok"
    message: str = "AMR Prediction API is running."
    models_loaded: int = 0
