"""
FastAPI Backend Unit Tests
===========================

Verifies that all API endpoints respond correctly using FastAPI's
built-in TestClient (powered by httpx) — no running uvicorn needed.

Tests:
    1. GET  /          — Root endpoint returns service info
    2. GET  /api/health  — Liveness probe returns 200 + model count
    3. GET  /api/models  — Returns list of loaded model summaries
    4. POST /api/predict — Valid JSON prediction request
    5. POST /api/predict — Invalid species returns 422
    6. POST /api/predict/file — File upload prediction
"""

import os
import sys
import io
import pytest
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from fastapi_backend.main import app


# --- Fixtures ---

@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app (models load on startup)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_spectrum():
    """Generate a realistic-looking synthetic MALDI-TOF spectrum (6000 bins)."""
    rng = np.random.RandomState(42)
    x = np.linspace(2000, 20000, 6000)
    spectrum = rng.exponential(100, size=6000)
    # Add Gaussian peaks to simulate real protein signals
    for center in [3500, 5000, 7800, 10200, 15000]:
        spectrum += 500 * np.exp(-((x - center) ** 2) / (2 * 200**2))
    return spectrum.tolist()


# --- Test Cases ---

class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_service_name(self, client):
        data = client.get("/").json()
        assert "AMR" in data["service"]

    def test_root_lists_endpoints(self, client):
        data = client.get("/").json()
        assert "endpoints" in data
        assert "health" in data["endpoints"]
        assert "models" in data["endpoints"]
        assert "predict_json" in data["endpoints"]


class TestHealthEndpoint:
    """Tests for GET /api/health"""

    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_health_models_loaded_count(self, client):
        data = client.get("/api/health").json()
        assert isinstance(data["models_loaded"], int)
        assert data["models_loaded"] >= 0


class TestModelsEndpoint:
    """Tests for GET /api/models"""

    def test_models_returns_200(self, client):
        response = client.get("/api/models")
        assert response.status_code == 200

    def test_models_returns_list(self, client):
        data = client.get("/api/models").json()
        assert "models" in data
        assert "count" in data
        assert isinstance(data["models"], list)

    def test_models_count_matches(self, client):
        data = client.get("/api/models").json()
        assert data["count"] == len(data["models"])

    def test_model_summary_fields(self, client):
        data = client.get("/api/models").json()
        if data["count"] > 0:
            model = data["models"][0]
            assert "species" in model
            assert "antibiotic" in model
            assert "clinical_name" in model
            assert "model_name" in model
            assert "cv_metrics" in model


class TestPredictEndpoint:
    """Tests for POST /api/predict"""

    def test_predict_valid_species(self, client, sample_spectrum):
        """Test prediction with a valid request body."""
        payload = {
            "species": "Staphylococcus aureus",
            "spectrum": sample_spectrum,
            "workstation": "ICU",
            "acquisition_date": "2026-05-28",
            "year": 2018,
        }
        response = client.post("/api/predict", json=payload)

        # 200 if models loaded, 404 if not
        assert response.status_code in (200, 404)

        if response.status_code == 200:
            data = response.json()
            assert data["species"] == "Staphylococcus aureus"
            assert data["prediction_label"] in ("Resistant", "Susceptible")
            assert 0.0 <= data["probability"] <= 1.0
            assert "clinical_report" in data
            assert "cv_metrics" in data
            assert "patient_metadata" in data
            assert data["patient_metadata"]["workstation"] == "ICU"

    def test_predict_invalid_species(self, client, sample_spectrum):
        """Invalid species should return 422 validation error."""
        payload = {
            "species": "Pseudomonas aeruginosa",
            "spectrum": sample_spectrum,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422

    def test_predict_wrong_spectrum_length(self, client):
        """Spectrum with wrong number of bins should return 422."""
        payload = {
            "species": "Staphylococcus aureus",
            "spectrum": [1.0] * 100,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422

    def test_predict_e_coli(self, client, sample_spectrum):
        """Test E. coli prediction."""
        payload = {
            "species": "Escherichia coli",
            "spectrum": sample_spectrum,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code in (200, 404)

    def test_predict_k_pneumoniae(self, client, sample_spectrum):
        """Test K. pneumoniae prediction."""
        payload = {
            "species": "Klebsiella pneumoniae",
            "spectrum": sample_spectrum,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code in (200, 404)


class TestPredictFileEndpoint:
    """Tests for POST /api/predict/file"""

    def test_predict_file_upload(self, client, sample_spectrum):
        """Test file upload prediction with a synthetic .txt spectrum."""
        lines = ["mz\tbinned_intensity\n"]
        for i, val in enumerate(sample_spectrum):
            lines.append(f"{2000 + i * 3.0:.1f}\t{val:.6f}\n")
        content = "".join(lines)

        response = client.post(
            "/api/predict/file",
            files={"file": ("test_spectrum.txt", io.BytesIO(content.encode()), "text/plain")},
            data={
                "species": "Staphylococcus aureus",
                "workstation": "ER",
                "acquisition_date": "2026-05-28",
                "year": 2018,
            },
        )

        assert response.status_code in (200, 404)

    def test_predict_file_invalid_species(self, client, sample_spectrum):
        """File upload with invalid species should return 400."""
        lines = ["mz\tbinned_intensity\n"]
        for i, val in enumerate(sample_spectrum):
            lines.append(f"{2000 + i * 3.0:.1f}\t{val:.6f}\n")
        content = "".join(lines)

        response = client.post(
            "/api/predict/file",
            files={"file": ("test.txt", io.BytesIO(content.encode()), "text/plain")},
            data={"species": "InvalidSpecies"},
        )
        assert response.status_code == 400

    def test_predict_file_bad_format(self, client):
        """File with missing binned_intensity column should return 400."""
        content = "col_a\tcol_b\n1.0\t2.0\n"
        response = client.post(
            "/api/predict/file",
            files={"file": ("bad.txt", io.BytesIO(content.encode()), "text/plain")},
            data={"species": "Staphylococcus aureus"},
        )
        assert response.status_code == 400
