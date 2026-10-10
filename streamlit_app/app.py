"""
Streamlit Clinical Dashboard Application
==========================================

Main entry point for the Streamlit web interface. Provides a premium
clinical dashboard for antimicrobial resistance (AMR) prediction from
MALDI-TOF mass spectrometry data.

Layout:
    - Sidebar: Species selector, patient metadata inputs, server status
    - Tab 1 — Prediction Hub: Upload/generate spectra, run inference, view results
    - Tab 2 — Model Performance: Live model stats from the FastAPI backend
    - Tab 3 — EDA Reference: Pre-computed figures from the preprocessing phase

Usage:
    # Start FastAPI backend first:
    .venv/bin/uvicorn fastapi_backend.main:app --port 8000

    # Then launch Streamlit:
    .venv/bin/streamlit run streamlit_app/app.py --server.port 8501
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env from the directory this file lives in (frontend/.env or root .env)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()  # also load root-level .env as fallback

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from streamlit_app.components.dashboard import (
    inject_custom_css,
    render_header,
    render_spectrum_plot,
    render_prediction_result,
    render_probability_bar,
    render_metrics_grid,
    render_clinical_report,
    render_server_status,
    render_eda_figure,
    render_imbalance_warning,
)

# --- Constants ---
# BACKEND_URL is read from the environment so the same code works both
# locally (pointing at localhost) and on Render (pointing at the deployed
# FastAPI service URL). Set it in frontend/.env or as a Render env var.
API_BASE_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# Allow overriding the EDA figures directory (useful when running from a
# different working directory, e.g. inside the frontend/ sub-folder).
FIGURES_DIR = os.environ.get(
    "FIGURES_DIR",
    os.path.join(PROJECT_ROOT, "notebooks", "figures"),
)

SPECIES_OPTIONS = {
    "Staphylococcus aureus": {
        "icon": "",
        "clinical": "MRSA Detection",
        "antibiotic": "Oxacillin",
        "description": "Methicillin-Resistant S. aureus — Beta-lactam resistance screening",
    },
    "Klebsiella pneumoniae": {
        "icon": "",
        "clinical": "Carbapenem Resistance",
        "antibiotic": "Meropenem",
        "description": "Carbapenem-Resistant K. pneumoniae — Critical biosafety threat",
    },
    "Escherichia coli": {
        "icon": "",
        "clinical": "3GC / ESBL Resistance",
        "antibiotic": "Ceftriaxone",
        "description": "Extended-Spectrum Beta-Lactamase — 3rd-gen cephalosporin resistance",
    },
}

WORKSTATION_OPTIONS = [
    "General Ward",
    "Intensive Care Unit (ICU)",
    "Emergency Room (ER)",
    "Surgical Ward",
    "Outpatient Clinic",
    "Blood Culture Lab",
    "Respiratory Ward",
    "Urology",
    "Neonatal ICU (NICU)",
]


# --- Page Configuration ---

st.set_page_config(
    page_title="AMR Diagnostics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


# --- Helper Functions ---

@st.cache_data(ttl=10, show_spinner=False)
def check_api_health() -> dict | None:
    """
    Ping the FastAPI /api/health endpoint.

    Cached for 10 seconds — short enough to reflect a server restart
    within a few re-runs, long enough to avoid hammering the backend
    on every widget interaction.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models_info() -> list | None:
    """
    Fetch model summaries from the FastAPI /api/models endpoint.

    Cached for 5 minutes — model metadata (CV metrics, thresholds) is
    static for the lifetime of a server process, so there is no reason
    to re-fetch it on every widget re-run.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/api/models", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass
    return None


def run_prediction_json(species: str, spectrum: list, workstation: str, date: str, year: int) -> dict | None:
    """Send a JSON prediction request to the FastAPI backend."""
    try:
        payload = {
            "species": species,
            "spectrum": spectrum,
            "workstation": workstation,
            "acquisition_date": date,
            "year": year,
        }
        # 180s timeout: accounts for Render free-tier cold start (~50s)
        # + ML inference + Gemini RAG report generation
        resp = requests.post(f"{API_BASE_URL}/api/predict", json=payload, timeout=180)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
    except requests.exceptions.Timeout:
        st.error(
            "⏱️ Request timed out. The backend may be waking up from sleep (Render free tier). "
            "Please visit the [health check](%s/api/health) URL to wake it, then try again." % API_BASE_URL
        )
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the FastAPI backend. Please ensure it is running.")
    except Exception as exc:
        st.error(f"❌ Request failed: {exc}")
    return None


def run_prediction_file(file_bytes: bytes, filename: str, species: str, workstation: str, date: str, year: int) -> dict | None:
    """Send a file upload prediction request to the FastAPI backend."""
    try:
        files = {"file": (filename, io.BytesIO(file_bytes), "text/plain")}
        data = {
            "species": species,
            "workstation": workstation,
            "acquisition_date": date,
            "year": str(year),
        }
        # 180s timeout: accounts for Render free-tier cold start (~50s)
        # + ML inference + Gemini RAG report generation
        resp = requests.post(f"{API_BASE_URL}/api/predict/file", files=files, data=data, timeout=180)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
    except requests.exceptions.Timeout:
        st.error(
            "⏱️ Request timed out. The backend may be waking up from sleep (Render free tier). "
            "Please visit the [health check](%s/api/health) URL to wake it, then try again." % API_BASE_URL
        )
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the FastAPI backend. Please ensure it is running.")
    except Exception as exc:
        st.error(f"❌ Request failed: {exc}")
    return None


def generate_synthetic_spectrum(seed: int = 42) -> np.ndarray:
    """
    Build a fake MALDI-TOF spectrum so someone can click through the app
    without a real data file handy.

    IMPORTANT: this is a DEMO CONVENIENCE ONLY. The peaks are arbitrary
    Gaussians on a random baseline — they carry no real resistance signal and
    are not tied to any true S/R label. Whatever the model predicts for a
    synthetic spectrum is meaningless; it only proves the pipeline runs. For a
    genuine test, upload one of the real DRIAMS spectra in test_spectra/.
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(2000, 20000, 6000)

    # Baseline noise (exponential decay + random)
    spectrum = rng.exponential(50, size=6000) + rng.normal(0, 10, size=6000)
    spectrum = np.maximum(spectrum, 0)

    # Add protein peaks at biologically relevant m/z positions
    peaks = [
        (3400, 200, 800),   # Small ribosomal protein
        (4600, 250, 1200),  # Hypothetical protein
        (5100, 180, 600),   # Cold shock protein
        (6300, 300, 1500),  # DNA-binding protein
        (7600, 220, 900),   # Stress response protein
        (9200, 350, 700),   # Outer membrane protein
        (10400, 280, 1100), # Translation factor
        (12500, 400, 500),  # Chaperonin subunit
        (15200, 500, 400),  # Large ribosomal protein
    ]
    for center, width, height in peaks:
        spectrum += height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

    return spectrum


# --- Sidebar ---

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Server Status
    health = check_api_health()
    if health:
        render_server_status(True, health.get("models_loaded", 0))
    else:
        render_server_status(False)

    st.markdown("")

    # Species Selector
    st.markdown("#### Target Pathogen")
    selected_species = st.selectbox(
        "Species",
        options=list(SPECIES_OPTIONS.keys()),
        format_func=lambda s: s,
        label_visibility="collapsed",
    )

    # Clear stale prediction results when the user switches species,
    # so old results from a different model are never shown alongside
    # a newly uploaded spectrum.
    if selected_species != st.session_state.get("_prev_species"):
        st.session_state["_prev_species"] = selected_species
        st.session_state.pop("last_result", None)

    info = SPECIES_OPTIONS[selected_species]
    st.caption(f"**{info['clinical']}** — {info['antibiotic']}")
    st.caption(info["description"])

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Patient Metadata
    st.markdown("#### Patient Metadata")
    workstation = st.selectbox("Ward / Workstation", WORKSTATION_OPTIONS)
    acq_date = st.date_input("Acquisition Date").strftime("%Y-%m-%d")
    acq_year = st.number_input("Year", min_value=2010, max_value=2030, value=2018, step=1)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # About Section
    st.markdown("#### About")
    st.caption(
        "This dashboard predicts antimicrobial resistance from "
        "MALDI-TOF mass spectrometry data using machine learning "
        "classifiers trained on the DRIAMS-A dataset (4 years, "
        "~11,500 samples). Clinical reports are generated via "
        "Retrieval-Augmented Generation (RAG) with Gemini."
    )


# --- Main Content ---

render_header()

tab1, tab2, tab3 = st.tabs([
    "Prediction Hub",
    "Model Performance",
    "EDA Reference",
])


# Tab 1: Prediction Hub
with tab1:
    st.markdown("### Select Resistance Test")
    st.caption("Choose which bacteria–antibiotic resistance you want to predict.")

    # --- Primary target selection dropdown (drives everything below) ---
    TARGET_OPTIONS = {
        "Staphylococcus aureus vs Oxacillin  (MRSA)": {
            "species": "Staphylococcus aureus",
            "antibiotic": "Oxacillin",
            "tag": "MRSA Detection",
        },
        "Klebsiella pneumoniae vs Meropenem  (CRKP)": {
            "species": "Klebsiella pneumoniae",
            "antibiotic": "Meropenem",
            "tag": "Carbapenem Resistance",
        },
        "Escherichia coli vs Ceftriaxone  (ESBL)": {
            "species": "Escherichia coli",
            "antibiotic": "Ceftriaxone",
            "tag": "ESBL Detection",
        },
    }

    chosen_target = st.selectbox(
        "Resistance Test",
        options=list(TARGET_OPTIONS.keys()),
        label_visibility="collapsed",
    )

    target = TARGET_OPTIONS[chosen_target]
    target_species = target["species"]
    target_antibiotic = target["antibiotic"]

    st.info(f"**{target['tag']}** — Model will evaluate *{target_species}* resistance to **{target_antibiotic}**.")

    # Clear stale results whenever the user switches the target so old
    # predictions from a different model are never shown.
    if chosen_target != st.session_state.get("_prev_target"):
        st.session_state["_prev_target"] = chosen_target
        st.session_state.pop("last_result", None)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # --- Spectrum Input Section ---
    st.markdown("### Spectrum Input")
    st.caption("Upload a DRIAMS-format `.txt` spectrum file or generate a synthetic demo spectrum.")

    if "spectrum_data" not in st.session_state:
        st.session_state["spectrum_data"] = None
    if "spectrum_source" not in st.session_state:
        st.session_state["spectrum_source"] = None
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state["uploaded_file_bytes"] = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None

    input_col1, input_col2 = st.columns([1, 1])

    with input_col1:
        st.markdown("##### Upload Spectrum File")
        uploaded_file = st.file_uploader(
            "Upload a .txt spectrum file",
            type=["txt", "csv", "tsv"],
            help="Tab/space-separated file with a `binned_intensity` column (6,000 rows).",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                text = file_bytes.decode("utf-8")
                df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=0)

                if "binned_intensity" in df.columns:
                    st.session_state["spectrum_data"] = df["binned_intensity"].values.astype(float)
                    st.session_state["spectrum_source"] = "file"
                    st.session_state["uploaded_file_bytes"] = file_bytes
                    st.session_state["uploaded_file_name"] = uploaded_file.name
                    # Clear previous prediction so stale results never linger.
                    st.session_state.pop("last_result", None)
                    st.success(f"Loaded {len(st.session_state['spectrum_data'])} bins from `{uploaded_file.name}`")
                else:
                    st.error(f"Missing `binned_intensity` column. Found: {list(df.columns)}")
            except Exception as exc:
                st.error(f"Failed to parse file: {exc}")

    with input_col2:
        st.markdown("##### Generate Synthetic Spectrum")
        seed = st.slider("Random Seed", 1, 100, 42, help="Change the seed to generate different spectral patterns.")
        if st.button("Generate Demo Spectrum", use_container_width=True):
            st.session_state["spectrum_data"] = generate_synthetic_spectrum(seed=seed)
            st.session_state["spectrum_source"] = "synthetic"
            st.session_state["uploaded_file_bytes"] = None
            st.session_state["uploaded_file_name"] = None
            st.session_state.pop("last_result", None)
            st.rerun()

    spectrum_data = st.session_state.get("spectrum_data")
    spectrum_source = st.session_state.get("spectrum_source")

    if spectrum_data is not None and len(spectrum_data) > 0:
        source_label = 'Synthetic demo' if spectrum_source == 'synthetic' else 'Uploaded file'
        st.success(f"Spectrum loaded: **{len(spectrum_data)}** bins ({source_label})")
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        chart_title = f"Input Spectrum — {target_species} vs {target_antibiotic}"
        render_spectrum_plot(spectrum_data, title=chart_title)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        if not health:
            st.warning("The FastAPI backend is offline. Please start the server on port 8000 before running predictions.")
        else:
            if st.button("Run AMR Prediction", type="primary", use_container_width=True):
                with st.spinner("Running inference pipeline... (TIC → PCA → Classify → RAG Report)"):
                    if spectrum_source == "file" and st.session_state.get("uploaded_file_bytes") is not None:
                        result = run_prediction_file(
                            file_bytes=st.session_state["uploaded_file_bytes"],
                            filename=st.session_state.get("uploaded_file_name", "spectrum.txt"),
                            species=target_species,
                            workstation=workstation,
                            date=acq_date,
                            year=acq_year,
                        )
                    else:
                        result = run_prediction_json(
                            species=target_species,
                            spectrum=spectrum_data.tolist(),
                            workstation=workstation,
                            date=acq_date,
                            year=acq_year,
                        )

                if result:
                    st.session_state["last_result"] = result

            if "last_result" in st.session_state:
                result = st.session_state["last_result"]

                render_prediction_result(
                    prediction_label=result["prediction_label"],
                    probability=result["probability"],
                    species=result["species"],
                    antibiotic=result["antibiotic"],
                )

                render_probability_bar(result["probability"])
                render_imbalance_warning(result["species"], result.get("cv_metrics", {}))

                st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
                st.markdown("### Model Validation Metrics")
                render_metrics_grid(result.get("cv_metrics", {}), model_name=result.get("model_name", "Unknown"))

                render_clinical_report(result.get("clinical_report", "*No report generated.*"))


# Tab 2: Model Performance Catalog
with tab2:
    st.markdown("### Active Model Performance Catalog")
    st.caption("Live statistics fetched from the FastAPI backend for all deployed AMR classifiers.")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    if not health:
        st.warning("⚠️ Cannot fetch model data — FastAPI backend is offline.")
    else:
        models_info = fetch_models_info()

        if models_info:
            for i, model in enumerate(models_info):
                species = model.get("species", "Unknown")
                antibiotic = model.get("antibiotic", "Unknown")
                clinical_name = model.get("clinical_name", "")
                model_name = model.get("model_name", "Unknown")
                cv = model.get("cv_metrics", {})

                st.markdown(f"""
                <div style="
                    background: rgba(22, 27, 34, 0.85);
                    border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px;
                    padding: 20px 24px 8px 24px;
                    margin-bottom: 8px;
                ">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 1.05rem; font-weight: 700; color: #e6edf3;">
                                {species}
                            </div>
                            <div style="font-size: 0.78rem; color: #8b949e;">
                                {clinical_name} · Target: <strong>{antibiotic}</strong>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                render_metrics_grid(cv, model_name=model_name)
                render_imbalance_warning(species, cv)

                if i < len(models_info) - 1:
                    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

            # Summary table
            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Summary Comparison Table")

            table_rows = []
            for m in models_info:
                cv = m.get("cv_metrics", {})
                table_rows.append({
                    "Species": m.get("species", ""),
                    "Antibiotic": m.get("antibiotic", ""),
                    "Model": m.get("model_name", ""),
                    "Threshold": f"{cv.get('threshold', 0.5):.2f}",
                    "ROC AUC": f"{cv.get('roc_auc', 0):.3f}",
                    "F1-Score": f"{cv.get('f1', 0):.3f}",
                    "Precision": f"{cv.get('precision', 0):.3f}",
                    "Recall": f"{cv.get('recall', 0):.3f}",
                    "Major Error (FPR)": f"{cv.get('fpr', 0):.1%}",
                    "V. Major Error (FNR)": f"{cv.get('fnr', 0):.1%}",
                })

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )

            # Clinical error explanations
            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Understanding Clinical Error Rates")

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label" style="color: #ffa502;">MAJOR ERROR (FALSE POSITIVE RATE)</div>
                    <div style="font-size: 0.85rem; color: #e6edf3; margin-top: 8px;">
                        The model predicts <strong>Resistant</strong> when the isolate is actually
                        <strong>Susceptible</strong>. This leads to <em>unnecessary escalation</em>
                        of antibiotic therapy — using stronger, more toxic, and more expensive
                        drugs when standard treatment would have worked.
                    </div>
                    <div style="font-size: 0.78rem; color: #8b949e; margin-top: 8px;">
                        Acceptable threshold: &lt; 3% (EUCAST/CLSI recommendation)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with exp_col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label" style="color: #ff4757;">VERY MAJOR ERROR (FALSE NEGATIVE RATE)</div>
                    <div style="font-size: 0.85rem; color: #e6edf3; margin-top: 8px;">
                        The model predicts <strong>Susceptible</strong> when the isolate is actually
                        <strong>Resistant</strong>. This is the most dangerous clinical scenario —
                        the patient receives <em>ineffective treatment</em>, potentially leading to
                        treatment failure, septic progression, or death.
                    </div>
                    <div style="font-size: 0.78rem; color: #8b949e; margin-top: 8px;">
                        Acceptable threshold: &lt; 1.5% (EUCAST/CLSI recommendation)
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("No models found. Ensure model training has been completed.")


# Tab 3: Historical EDA Reference
with tab3:
    st.markdown("### Exploratory Data Analysis — DRIAMS-A Dataset")
    st.caption(
        "Pre-computed visualizations from the preprocessing phase. "
        "These figures summarize the raw dataset characteristics, class distributions, "
        "and dimensionality reduction results across all target pathogens."
    )
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Dataset Overview
    st.markdown("#### Dataset Statistics")
    overview_cols = st.columns(3)
    with overview_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">S. AUREUS / OXACILLIN</div>
            <div class="metric-value" style="color: #00d4aa;">3,790</div>
            <div class="metric-detail">Samples · 80.8% S / 19.2% R · 1,451 PCs</div>
        </div>
        """, unsafe_allow_html=True)
    with overview_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">K. PNEUMONIAE / MEROPENEM</div>
            <div class="metric-value" style="color: #ffa502;">2,818</div>
            <div class="metric-detail">Samples · 99.1% S / 0.9% R · 1,249 PCs</div>
        </div>
        """, unsafe_allow_html=True)
    with overview_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">E. COLI / CEFTRIAXONE</div>
            <div class="metric-value" style="color: #00d4aa;">4,960</div>
            <div class="metric-detail">Samples · 78.1% S / 21.9% R · 1,550 PCs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Class Distribution
    st.markdown("#### Class Distribution")
    render_eda_figure(
        os.path.join(FIGURES_DIR, "01_class_distribution.png"),
        caption="Binary label distribution (Susceptible vs Resistant) across all three target pathogen-antibiotic pairs."
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Mean Spectra
    st.markdown("#### Mean Spectra Overlay")
    render_eda_figure(
        os.path.join(FIGURES_DIR, "02_mean_spectra_overlay.png"),
        caption="Average MALDI-TOF spectra for Susceptible (blue) vs Resistant (red) isolates across 6,000 m/z bins."
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Variance Distribution
    st.markdown("#### Feature Variance Distribution")
    render_eda_figure(
        os.path.join(FIGURES_DIR, "03_variance_distribution.png"),
        caption="Distribution of per-feature variance across all 6,000 spectral bins (log scale). Low-variance bins are removed by the preprocessing pipeline."
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # PCA Scatter Plots
    st.markdown("#### PCA 2D Projections")
    st.caption("Principal Component Analysis scatter plots showing how spectra cluster in reduced-dimensionality space.")

    pca_cols = st.columns(3)
    pca_species = [
        ("staphylococcus_aureus", "S. aureus (Oxacillin)"),
        ("klebsiella_pneumoniae", "K. pneumoniae (Meropenem)"),
        ("escherichia_coli", "E. coli (Ceftriaxone)"),
    ]
    for col, (sp_key, sp_label) in zip(pca_cols, pca_species):
        with col:
            render_eda_figure(
                os.path.join(FIGURES_DIR, f"06_pca_2d_{sp_key}.png"),
                caption=sp_label,
            )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # PCA Explained Variance
    st.markdown("#### PCA Cumulative Explained Variance")
    st.caption("Elbow curves showing the number of principal components required to reach 95% explained variance.")

    pca_var_cols = st.columns(3)
    for col, (sp_key, sp_label) in zip(pca_var_cols, pca_species):
        with col:
            render_eda_figure(
                os.path.join(FIGURES_DIR, f"05_pca_variance_{sp_key}.png"),
                caption=sp_label,
            )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Correlation Heatmaps
    st.markdown("#### Top Feature Correlations")
    st.caption("Correlation heatmaps of the top-variance spectral bins for each species.")

    corr_cols = st.columns(3)
    for col, (sp_key, sp_label) in zip(corr_cols, pca_species):
        with col:
            render_eda_figure(
                os.path.join(FIGURES_DIR, f"04_correlation_{sp_key}.png"),
                caption=sp_label,
            )
