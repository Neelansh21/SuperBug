"""
Streamlit Dashboard UI Components
===================================

Library of reusable rendering functions for the clinical dashboard.
Each function handles a single visual element:
- Custom CSS injection (dark theme, glassmorphism cards, typography)
- Header branding
- Interactive Plotly spectrum visualization
- Color-coded prediction result cards
- Cross-validation metrics grid
- RAG clinical report renderer
"""

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# --- 1. Custom CSS Injection ---

def inject_custom_css():
    """
    Inject a comprehensive custom CSS stylesheet into the Streamlit session.

    Transforms the default Streamlit appearance into a premium dark-mode
    clinical dashboard with charcoal backgrounds, neon teal accents,
    glassmorphism effects, Inter font, and smooth micro-animations.
    """
    st.markdown("""
    <style>
        /* ─── Google Fonts ─── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ─── Root Variables ─── */
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #161b22;
            --bg-card: rgba(22, 27, 34, 0.85);
            --accent-teal: #00d4aa;
            --accent-teal-dim: rgba(0, 212, 170, 0.15);
            --accent-red: #ff4757;
            --accent-red-dim: rgba(255, 71, 87, 0.15);
            --accent-amber: #ffa502;
            --accent-amber-dim: rgba(255, 165, 2, 0.15);
            --accent-blue: #3742fa;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-accent: rgba(0, 212, 170, 0.3);
            --shadow-glow: 0 0 20px rgba(0, 212, 170, 0.08);
            --radius: 12px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ─── Global Font ─── */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* ─── Scrollbar Styling ─── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-teal); }

        /* ─── Sidebar ─── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1117 0%, #101820 100%) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--accent-teal) !important;
        }

        /* ─── Tab Styling ─── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--bg-secondary);
            border-radius: var(--radius);
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.9rem;
            padding: 8px 20px;
            transition: var(--transition);
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-teal-dim) !important;
            color: var(--accent-teal) !important;
            border-bottom-color: var(--accent-teal) !important;
        }

        /* ─── Metric Card ─── */
        .metric-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius);
            padding: 20px 24px;
            margin-bottom: 12px;
            box-shadow: var(--shadow-glow);
            transition: var(--transition);
        }
        .metric-card:hover {
            border-color: var(--border-accent);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 212, 170, 0.12);
        }
        .metric-card .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .metric-card .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.1;
        }
        .metric-card .metric-detail {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        /* ─── Status Badges ─── */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .status-badge.online {
            background: rgba(0, 212, 170, 0.12);
            color: #00d4aa;
            border: 1px solid rgba(0, 212, 170, 0.25);
        }
        .status-badge.offline {
            background: rgba(255, 71, 87, 0.12);
            color: #ff4757;
            border: 1px solid rgba(255, 71, 87, 0.25);
        }

        /* ─── Result Cards (Resistant / Susceptible) ─── */
        .result-card {
            border-radius: var(--radius);
            padding: 28px 32px;
            margin: 16px 0;
            text-align: center;
        }
        .result-card.resistant {
            background: linear-gradient(135deg, rgba(255, 71, 87, 0.08) 0%, rgba(255, 71, 87, 0.02) 100%);
            border: 1.5px solid rgba(255, 71, 87, 0.35);
        }
        .result-card.susceptible {
            background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(0, 212, 170, 0.02) 100%);
            border: 1.5px solid rgba(0, 212, 170, 0.35);
        }
        .result-card .result-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 8px;
        }
        .result-card.resistant .result-label { color: var(--accent-red); }
        .result-card.susceptible .result-label { color: var(--accent-teal); }
        .result-card .result-value {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .result-card.resistant .result-value { color: var(--accent-red); }
        .result-card.susceptible .result-value { color: var(--accent-teal); }
        .result-card .result-prob {
            font-size: 1rem;
            color: var(--text-secondary);
        }

        /* ─── Divider ─── */
        .styled-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-accent), transparent);
            margin: 24px 0;
            border: none;
        }

        /* ─── Report Container ─── */
        .report-container {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius);
            padding: 28px 32px;
            margin-top: 16px;
            box-shadow: var(--shadow-glow);
        }

        /* ─── Warning Banner ─── */
        .warning-banner {
            background: var(--accent-amber-dim);
            border: 1px solid rgba(255, 165, 2, 0.3);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.82rem;
            color: var(--accent-amber);
            margin-bottom: 12px;
        }

        /* ─── Animated Pulse Dot ─── */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .pulse-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 1.5s ease-in-out infinite;
        }
        .pulse-dot.green { background: #00d4aa; }
        .pulse-dot.red { background: #ff4757; }

        /* ─── Hide Streamlit Defaults ─── */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# --- 2. Header ---

def render_header():
    """Render the dashboard header with title and subtitle."""
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 20px 0;">
        <h1 style="
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 50%, #3742fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 4px;
            letter-spacing: -0.5px;
        ">AMR Diagnostics Dashboard</h1>
        <p style="
            font-size: 0.88rem;
            color: #8b949e;
            font-weight: 400;
            letter-spacing: 0.3px;
        ">MALDI-TOF Mass Spectrometry · ML Classification · RAG Clinical Reports</p>
    </div>
    <div class="styled-divider"></div>
    """, unsafe_allow_html=True)


# --- 3. Interactive Spectrum Plot (Plotly) ---

def render_spectrum_plot(raw_spectrum: list | np.ndarray, title: str = "MALDI-TOF Mass Spectrum"):
    """
    Render an interactive Plotly line chart of a 6,000-bin MALDI-TOF spectrum.

    Plotly is used over matplotlib because it allows hover inspection of
    individual m/z bins, zoom into protein peak regions, and full panning.
    """
    if isinstance(raw_spectrum, list):
        raw_spectrum = np.array(raw_spectrum)

    mz_values = np.linspace(2000, 20000, len(raw_spectrum))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mz_values,
        y=raw_spectrum,
        mode="lines",
        line=dict(color="#00d4aa", width=1.2),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 170, 0.06)",
        hovertemplate="<b>m/z:</b> %{x:.1f}<br><b>Intensity:</b> %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Inter", size=14, color="#8b949e"),
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        margin=dict(l=50, r=20, t=40, b=50),
        height=340,
        hovermode="x unified",
    )

    # Plotly 6.x: axis styling must use update_xaxes / update_yaxes
    fig.update_xaxes(
        title_text="m/z (Da)",
        title_font=dict(family="Inter", size=12, color="#8b949e"),
        tickfont=dict(family="Inter", size=11, color="#6e7681"),
        gridcolor="rgba(255,255,255,0.04)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.06)",
    )
    fig.update_yaxes(
        title_text="Intensity",
        title_font=dict(family="Inter", size=12, color="#8b949e"),
        tickfont=dict(family="Inter", size=11, color="#6e7681"),
        gridcolor="rgba(255,255,255,0.04)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.06)",
    )

    st.plotly_chart(fig, use_container_width=True)


# --- 4. Prediction Result Card ---

def render_prediction_result(prediction_label: str, probability: float, species: str, antibiotic: str):
    """
    Render a large, color-coded prediction result card.
    Red gradient for Resistant, teal gradient for Susceptible.
    """
    css_class = "resistant" if prediction_label == "Resistant" else "susceptible"
    confidence = "HIGH" if (probability > 0.85 or probability < 0.15) else "MODERATE"

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label">AMR Prediction Result</div>
        <div class="result-value">{prediction_label.upper()}</div>
        <div class="result-prob">
            {probability * 100:.1f}% probability of resistance to <strong>{antibiotic}</strong>
            &nbsp;·&nbsp; Confidence: <strong>{confidence}</strong>
        </div>
        <div style="margin-top: 8px; font-size: 0.78rem; color: #6e7681;">
            Species: <em>{species}</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- 5. Probability Progress Bar ---

def render_probability_bar(probability: float):
    """Render a horizontal gradient bar showing resistance probability."""
    pct = probability * 100
    # Color transitions: green (0%) → amber (50%) → red (100%)
    if pct < 40:
        bar_color = "#00d4aa"
    elif pct < 60:
        bar_color = "#ffa502"
    else:
        bar_color = "#ff4757"

    st.markdown(f"""
    <div style="margin: 12px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 0.75rem; color: #00d4aa; font-weight: 600;">SUSCEPTIBLE</span>
            <span style="font-size: 0.75rem; color: #ff4757; font-weight: 600;">RESISTANT</span>
        </div>
        <div style="
            background: rgba(255,255,255,0.06);
            border-radius: 6px;
            height: 10px;
            overflow: hidden;
        ">
            <div style="
                width: {pct:.1f}%;
                height: 100%;
                background: linear-gradient(90deg, #00d4aa, {bar_color});
                border-radius: 6px;
                transition: width 0.8s ease;
            "></div>
        </div>
        <div style="text-align: center; margin-top: 4px; font-size: 0.78rem; color: #8b949e;">
            Resistance Probability: <strong>{pct:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- 6. Cross-Validation Metrics Grid ---

def render_metrics_grid(cv_metrics: dict, model_name: str = "Unknown"):
    """
    Render a grid of glassmorphism metric cards showing CV performance.

    Displays clinical error rates (Major Error / Very Major Error) alongside
    standard ML metrics because clinicians need to understand:
    - Major Error (FPR): Risk of unnecessary aggressive treatment
    - Very Major Error (FNR): Risk of treatment failure / fatal mistreatment
    """
    st.markdown(f"""
    <div style="
        text-align: center;
        margin-bottom: 16px;
        font-size: 0.82rem;
        color: #8b949e;
    ">
        Classifier: <strong style="color: #00d4aa;">{model_name}</strong>
        &nbsp;·&nbsp; 5-Fold Stratified Cross-Validation
        &nbsp;·&nbsp; Decision Threshold: <strong style="color: #ffa502;">{cv_metrics.get("threshold", 0.5):.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    roc = cv_metrics.get("roc_auc", 0)
    f1 = cv_metrics.get("f1", 0)
    prec = cv_metrics.get("precision", 0)
    rec = cv_metrics.get("recall", 0)
    fpr = cv_metrics.get("fpr", 0)
    fnr = cv_metrics.get("fnr", 0)
    avg_prec = cv_metrics.get("avg_prec", 0)
    threshold = cv_metrics.get("threshold", 0.5)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_single_metric("ROC AUC", f"{roc:.3f}", "Area Under Curve", _metric_color(roc, [0.7, 0.85]))
    with col2:
        _render_single_metric("F1-SCORE", f"{f1:.3f}", "Harmonic Mean", _metric_color(f1, [0.5, 0.7]))
    with col3:
        _render_single_metric("PRECISION", f"{prec:.3f}", "Positive Pred. Value", _metric_color(prec, [0.5, 0.7]))
    with col4:
        _render_single_metric("RECALL", f"{rec:.3f}", "Sensitivity", _metric_color(rec, [0.5, 0.7]))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        _render_single_metric("AVG PRECISION", f"{avg_prec:.3f}", "Precision-Recall AUC", _metric_color(avg_prec, [0.4, 0.7]))
    with col6:
        _render_single_metric("MAJOR ERROR", f"{fpr:.1%}", "FPR · Unnecessary Rx", _error_color(fpr, [0.03, 0.10]), is_error=True)
    with col7:
        _render_single_metric("V. MAJOR ERROR", f"{fnr:.1%}", "FNR · Treatment Failure", _error_color(fnr, [0.10, 0.30]), is_error=True)
    with col8:
        _render_single_metric("THRESHOLD", f"{threshold:.2f}", "Optimal Decision Cutoff", "#ffa502")


def _metric_color(value: float, thresholds: list) -> str:
    """Return green/amber/red based on metric value (higher is better)."""
    if value >= thresholds[1]:
        return "#00d4aa"
    elif value >= thresholds[0]:
        return "#ffa502"
    return "#ff4757"


def _error_color(value: float, thresholds: list) -> str:
    """Return green/amber/red based on error rate (lower is better)."""
    if value <= thresholds[0]:
        return "#00d4aa"
    elif value <= thresholds[1]:
        return "#ffa502"
    return "#ff4757"


def _render_single_metric(label: str, value: str, detail: str, color: str, is_error: bool = False):
    """Render one glassmorphism metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-detail">{detail}</div>
    </div>
    """, unsafe_allow_html=True)


# --- 7. Clinical Report Renderer ---

def render_clinical_report(report_markdown: str):
    """Render the RAG-generated clinical report inside a styled glass container."""
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    ">
        <span style="
            font-size: 1rem;
            font-weight: 700;
            color: #e6edf3;
            letter-spacing: -0.3px;
        ">Clinical Interpretation Report</span>
        <span style="
            font-size: 0.7rem;
            background: rgba(0, 212, 170, 0.12);
            color: #00d4aa;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 600;
        ">RAG-Generated</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(report_markdown)
    st.markdown('</div>', unsafe_allow_html=True)


# --- 8. Server Status Badge ---

def render_server_status(is_online: bool, models_loaded: int = 0):
    """Render a connection status badge in the sidebar."""
    if is_online:
        st.markdown(f"""
        <div class="status-badge online">
            <span class="pulse-dot green"></span>
            API Online · {models_loaded} models
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge offline">
            <span class="pulse-dot red"></span>
            API Offline
        </div>
        """, unsafe_allow_html=True)


# --- 9. EDA Figure Display ---

def render_eda_figure(figure_path: str, caption: str = ""):
    """Display a pre-generated EDA figure if it exists."""
    if os.path.exists(figure_path):
        st.image(figure_path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure not found: `{os.path.basename(figure_path)}`")


# --- 10. Imbalance Warning Banner ---

def render_imbalance_warning(species: str, metrics: dict):
    """
    Display a contextual warning for models trained on highly imbalanced data.

    Specifically targets the K. pneumoniae / Meropenem model (0.9% resistance
    prevalence) to alert clinicians about prediction limitations.
    """
    f1 = metrics.get("f1", 1.0)
    fnr = metrics.get("fnr", 0.0)

    if f1 < 0.2 or fnr > 0.5:
        st.markdown(f"""
        <div class="warning-banner">
            <strong>Clinical Data Limitation:</strong> The model for <em>{species}</em>
            was trained on a dataset with extreme class imbalance (very few resistant samples).
            The F1-score is <strong>{f1:.3f}</strong> and the Very Major Error rate is
            <strong>{fnr:.1%}</strong>. Use this prediction with caution and confirm with
            phenotypic antimicrobial susceptibility testing (AST).
        </div>
        """, unsafe_allow_html=True)
