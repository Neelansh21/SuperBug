"""
FastAPI Application Entry Point
=================================

Central application factory for the AMR Prediction API. Performs three
critical startup tasks:

1. CORS Middleware — Enables cross-origin requests from the Streamlit
   front-end (running on a different port).
2. Router Mounting — Registers the analysis router under ``/api/``.
3. Startup Model Loading — Pre-caches all species-specific preprocessing
   pipelines and trained ML models into memory for sub-second inference.

Run locally:
    uvicorn fastapi_backend.main:app --reload --port 8000
"""


import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment configuration from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi_backend.routers.analysis import router as analysis_router, load_all_models




@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Eagerly loads all serialized model packages and preprocessing pipelines
    into an in-memory cache so that every /predict request runs without
    touching the filesystem.
    """


    load_all_models()



    yield  # Application runs here




app = FastAPI(
    title="AMR Prediction API",
    description=(
        "Production-grade REST API for antimicrobial resistance prediction "
        "from MALDI-TOF mass spectrometry data. Powered by tree-based ML "
        "classifiers and LangChain RAG clinical report generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware — the browser blocks the Streamlit frontend (a different
# origin/port) from calling this API unless we explicitly allow it here.
# In production, set ALLOW_ORIGINS env var to your Streamlit URL, e.g.:
#   ALLOW_ORIGINS=https://amr-frontend.onrender.com
# For local dev or open access, leave ALLOW_ORIGINS=* (the default).
_raw_origins = os.getenv("ALLOW_ORIGINS", "*")
_allow_origins: list[str] = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api")


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "AMR Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "predict_json": "/api/predict",
            "predict_file": "/api/predict/file",
        },
    }
