"""
LangChain & RAG Unit Tests
===========================

Verifies that the clinical guidelines are parsed correctly, and that the
RAG retriever and report chains behave as expected.
"""

import os
import pytest
from langchain_engine.embeddings import ClinicalGuidelinesRetriever
from langchain_engine.chains import generate_clinical_report


def test_guidelines_parsing():
    """Verify that guidelines.md is parsed correctly into clinical sections."""
    retriever = ClinicalGuidelinesRetriever()

    assert len(retriever.sections) == 3

    titles = [sec["title"] for sec in retriever.sections]
    assert "MRSA (Staphylococcus aureus - Oxacillin Resistance)" in titles
    assert "CRKP (Klebsiella pneumoniae - Meropenem/Carbapenem Resistance)" in titles
    assert "ESBL (Escherichia coli - Ceftriaxone Resistance)" in titles


def test_retriever_keyword_matching():
    """Verify that keyword retrieval returns the correct section for a query."""
    retriever = ClinicalGuidelinesRetriever()

    # MRSA query should match S. aureus section
    results = retriever.retrieve("Staphylococcus aureus Oxacillin", k=1)
    assert len(results) == 1
    assert "MRSA" in results[0]["title"]

    # CRKP query should match K. pneumoniae section
    results = retriever.retrieve("Klebsiella pneumoniae Meropenem", k=1)
    assert len(results) == 1
    assert "CRKP" in results[0]["title"]

    # ESBL query should match E. coli section
    results = retriever.retrieve("Escherichia coli Ceftriaxone", k=1)
    assert len(results) == 1
    assert "ESBL" in results[0]["title"]


def test_generate_clinical_report_offline(monkeypatch):
    """Verify report compiling and structural integrity under local offline mode."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    metadata = {
        "workstation": "Blood Culture",
        "acquisition_date": "2026-05-28",
        "year": 2018,
    }

    # No API key → forces local mock report generation
    report = generate_clinical_report(
        species="Staphylococcus aureus",
        antibiotic="Oxacillin",
        prediction_prob=0.895,
        label=1,
        metadata=metadata,
        api_key=None,
    )

    # Verify the generated markdown report structure
    assert "# CLINICAL INTERPRETATION REPORT" in report
    assert "Staphylococcus aureus" in report
    assert "DEVELOPER MODE" in report  # confirms mock fallback was triggered
    assert "Executive Summary" in report
    assert "Resistant" in report
    assert "89.5%" in report
    assert "Therapeutic Actionability" in report
    assert "Infection Control" in report
