"""
LangChain Clinical Report Chain Orchestrator
=============================================

Orchestrates clinical report generation by retrieving relevant guidelines
from the RAG database, binding patient metadata, formatting the structured
prompt, and querying the ChatGoogleGenerativeAI (Gemini) LLM.

Offline Fallback:
    If the Gemini API key is missing or the connection fails, a structured
    template formatter compiles the actual RAG guidelines and prediction
    details into a clean markdown document. This keeps the web app fully
    interactive even in local offline environments.
"""

import os
import logging

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_engine.embeddings import ClinicalGuidelinesRetriever
from langchain_engine.prompts import CLINICAL_REPORT_PROMPT
from ml_pipeline.preprocessing import TARGET_CONFIG

logger = logging.getLogger(__name__)

# Load environment configuration from .env file
load_dotenv()




def _format_mock_clinical_report(
    species: str,
    antibiotic: str,
    prediction_prob: float,
    label: int,
    workstation: str,
    acquisition_date: str,
    year: str,
    clinical_name: str,
    guidelines_text: str,
) -> str:
    """
    Format a detailed clinical report offline when no API key is present.

    Compiles a report directly from the RAG guidelines so that the user
    gets a realistic demonstration document even without API keys.
    """
    pred_label = "Resistant" if label == 1 else "Susceptible"
    confidence = "HIGH" if (prediction_prob > 0.85 or prediction_prob < 0.15) else "MODERATE"

    report = f"""# CLINICAL INTERPRETATION REPORT: {species} ({clinical_name})

> [!NOTE]
> **DEVELOPER MODE**: This report was compiled locally using cached reference guidelines because the Gemini API key was not supplied.

## 1. Executive Summary
Diagnostic profiling suggests that this strain of ***{species}*** is **{pred_label}** to **{antibiotic}** with a prediction probability of **{prediction_prob * 100:.1f}%**. Clinical confidence is rated as **{confidence}** based on machine learning probability bounds.

## 2. Machine Learning Diagnostic Insight
*   **Methodology**: MALDI-TOF mass spectrometry profile analysis via non-linear classification models.
*   **Result**: {prediction_prob * 100:.1f}% probability of resistance to {antibiotic}.
*   **Diagnostic Implication**:
"""

    # Add pathogen-specific clinical context
    if "aureus" in species.lower():
        report += "    This pattern is indicative of Methicillin-Resistant *Staphylococcus aureus* (MRSA), signifying resistance to almost all standard beta-lactam antibiotics. Immediate alternative therapy is recommended if active infection is confirmed."
    elif "pneumoniae" in species.lower():
        report += "    This pattern indicates Carbapenem-Resistant *Klebsiella pneumoniae* (CRKP), representing a critical biosafety threat due to carbapenemase production. High risk of treatment failure; combination therapeutic regimens should be reviewed immediately."
    else:
        report += "    This pattern suggests Extended-Spectrum Beta-Lactamase (ESBL) producing *Escherichia coli*, conferring resistance to third-generation cephalosporins (e.g., ceftriaxone)."

    report += f"""

## 3. Therapeutic Actionability
The following guidelines have been retrieved from the RAG clinical reference database:

{guidelines_text}

## 4. Infection Control & Biosafety
*   **Precautions**: Check guideline instructions for {species} isolation protocols.
*   **Diagnostics**: Align this predictive screening result with confirmatory phenotypic AST (antimicrobial susceptibility testing) as soon as laboratory broth microdilution data becomes available.
*   **Infection Control**: Contact the clinical ward coordinator and alert infection control if a Resistant status is predicted.
"""
    return report


def generate_clinical_report(
    species: str,
    antibiotic: str,
    prediction_prob: float,
    label: int,
    metadata: dict,
    api_key: str = None,
) -> str:
    """
    Load guidelines, format prompt parameters, and generate a clinical report.

    Parameters
    ----------
    species : str
        Name of the pathogen species.
    antibiotic : str
        Tested antibiotic.
    prediction_prob : float
        ML probability score of resistance [0.0 - 1.0].
    label : int
        Binary prediction class (0: Susceptible, 1: Resistant).
    metadata : dict
        Patient sample metadata (acquisition date, year, workstation).
    api_key : str, optional
        API key to override environment settings.

    Returns
    -------
    str
        Markdown formatted clinical interpretation report.
    """
    # 1. Resolve API key
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # 2. Retrieve guidelines using RAG embeddings
    retriever = ClinicalGuidelinesRetriever(api_key=key)
    query = f"{species} {antibiotic}"
    docs = retriever.retrieve(query, k=1)

    guidelines_text = docs[0]["content"] if docs else "No guidelines found."

    # 3. Resolve metadata details
    workstation = metadata.get("workstation", "Unknown Workstation")
    acq_date = metadata.get("acquisition_date", "Unknown Date")
    year = str(metadata.get("year", "Unknown Year"))

    clinical_name = TARGET_CONFIG.get(species, {}).get("clinical_name", "AMR Diagnostic Report")
    pred_label = "Resistant" if label == 1 else "Susceptible"

    # 4. Invoke LLM if API key is present; otherwise fall back to structured formatter
    if not key:
        return _format_mock_clinical_report(
            species, antibiotic, prediction_prob, label, workstation, acq_date, year, clinical_name, guidelines_text
        )

    try:
        # Dynamic model selection with fallback to gemini-2.5-flash for reliability
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=key,
        )

        # Build chain using LCEL
        chain = CLINICAL_REPORT_PROMPT | llm | StrOutputParser()

        report = chain.invoke({
            "species": species,
            "antibiotic": antibiotic,
            "prediction_label": pred_label,
            "probability": prediction_prob * 100,
            "workstation": workstation,
            "acquisition_date": acq_date,
            "year": year,
            "guidelines": guidelines_text,
            "clinical_name": clinical_name,
        })
        return report

    except Exception as exc:
        # Safe fallback for rate-limiting, network dropouts, or credentials mismatch
        logger.warning("Gemini report generation failed, using offline fallback: %s", exc)
        return _format_mock_clinical_report(
            species, antibiotic, prediction_prob, label, workstation, acq_date, year, clinical_name, guidelines_text
        )
