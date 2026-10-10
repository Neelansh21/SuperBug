"""
Backend Entry Point
====================
Thin module that can be used as the uvicorn target when running from
the ``backend/`` directory (where sys.path includes the repo root).

Run locally:
    cd backend
    uvicorn main:app --reload --port 8000

Or from the repo root (original behaviour):
    uvicorn fastapi_backend.main:app --reload --port 8000
"""

# Re-export the FastAPI app from the package so uvicorn can target this file.
from fastapi_backend.main import app  # noqa: F401

__all__ = ["app"]
