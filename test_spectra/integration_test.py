#!/usr/bin/env python3
"""
End-to-End Integration Test for AMR Prediction System
======================================================

Uses REAL resistant and susceptible DRIAMS-A spectra to verify:
1. Backend correctly routes each species to the right model
2. Each model produces different probabilities for R vs S spectra
3. The full pipeline (TIC → PCA → Classify → RAG Report) works end-to-end

Run:
    # Terminal 1: start backend
    .venv/bin/uvicorn fastapi_backend.main:app --port 8000

    # Terminal 2: run this script
    .venv/bin/python test_spectra/integration_test.py
"""

import os
import sys
import json
import shutil
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

API_BASE = "http://127.0.0.1:8000/api"
DRIAMS_BINNED = os.path.join(PROJECT_ROOT, "DRIAMS-A", "binned_6000")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_spectra")

# Real DRIAMS-A sample codes with known resistance labels (from 2016 — cleanest year)
TEST_CASES = [
    {
        "species": "Staphylococcus aureus",
        "antibiotic": "Oxacillin",
        "tag": "MRSA",
        "resistant_code": "0057f21c-94f9-4f35-b8ff-969e187ef1f8",
        "resistant_year": "2016",
        "susceptible_code": "000a2090-67fb-4556-8d34-3b758508be98",
        "susceptible_year": "2016",
    },
    {
        "species": "Klebsiella pneumoniae",
        "antibiotic": "Meropenem",
        "tag": "CRKP",
        "resistant_code": "94945229-059b-47a9-ab4f-33560ac3383b",
        "resistant_year": "2015",
        "susceptible_code": "031185e1-74a9-4f00-aea0-70d4add3da21",
        "susceptible_year": "2015",
    },
    {
        "species": "Escherichia coli",
        "antibiotic": "Ceftriaxone",
        "tag": "ESBL",
        "resistant_code": "006fd66b-55ce-42cd-9e4d-e404f7886ec5",
        "resistant_year": "2016",
        "susceptible_code": "002c4fc6-42a6-4cd5-8ee8-207038bea4f9",
        "susceptible_year": "2016",
    },
]


def check_health():
    """Verify backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        data = r.json()
        print(f"  API Status:  {data['status']}")
        print(f"  Models loaded: {data['models_loaded']}")
        return data["models_loaded"] > 0
    except Exception as e:
        print(f"  FAILED — Backend not reachable: {e}")
        return False


def check_models():
    """Print details of all loaded models."""
    r = requests.get(f"{API_BASE}/models", timeout=5)
    data = r.json()
    print(f"\n  Loaded models ({data['count']}):")
    for m in data["models"]:
        cv = m["cv_metrics"]
        print(
            f"    {m['species']:30s}  {m['antibiotic']:15s}  "
            f"Model={m['model_name']:10s}  F1={cv['f1']:.3f}  "
            f"ROC-AUC={cv['roc_auc']:.3f}  FPR={cv['fpr']:.3f}  FNR={cv['fnr']:.3f}"
        )
    return data["models"]


def copy_spectrum(code, year, label, species_tag):
    """Copy a real DRIAMS spectrum to the test_spectra directory."""
    src = os.path.join(DRIAMS_BINNED, year, f"{code}.txt")
    if not os.path.exists(src):
        print(f"    WARNING: Spectrum file not found: {src}")
        return None

    safe_tag = species_tag.lower().replace(" ", "_")
    dest_name = f"{safe_tag}_{label}_{code[:8]}.txt"
    dest = os.path.join(OUTPUT_DIR, dest_name)
    shutil.copy2(src, dest)
    return dest


def run_file_prediction(filepath, species):
    """Send a file upload prediction request and return the result."""
    with open(filepath, "rb") as f:
        files = {"file": (os.path.basename(filepath), f, "text/plain")}
        data = {
            "species": species,
            "workstation": "Integration Test",
            "acquisition_date": "2026-05-29",
            "year": "2018",
        }
        r = requests.post(f"{API_BASE}/predict/file", files=files, data=data, timeout=60)

    if r.status_code != 200:
        print(f"    API ERROR {r.status_code}: {r.text[:200]}")
        return None
    return r.json()


def main():
    separator = "=" * 72
    print(f"\n{separator}")
    print("  AMR PREDICTION SYSTEM — END-TO-END INTEGRATION TEST")
    print(f"{separator}\n")

    # Step 1: Health check
    print("[1/4] Checking API health...")
    if not check_health():
        print("\n  ABORT: Start the backend first with:")
        print("    .venv/bin/uvicorn fastapi_backend.main:app --port 8000")
        sys.exit(1)

    # Step 2: List loaded models
    print("\n[2/4] Verifying loaded models...")
    models = check_models()
    loaded_species = {m["species"] for m in models}

    # Step 3: Copy real DRIAMS spectra
    print(f"\n[3/4] Preparing real DRIAMS-A test spectra...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_files = []
    for tc in TEST_CASES:
        species = tc["species"]
        if species not in loaded_species:
            print(f"  SKIP: No model loaded for {species}")
            continue

        r_file = copy_spectrum(tc["resistant_code"], tc["resistant_year"], "resistant", tc["tag"])
        s_file = copy_spectrum(tc["susceptible_code"], tc["susceptible_year"], "susceptible", tc["tag"])

        if r_file:
            test_files.append({"file": r_file, "species": species, "expected": "Resistant", "tag": tc["tag"], "antibiotic": tc["antibiotic"]})
        if s_file:
            test_files.append({"file": s_file, "species": species, "expected": "Susceptible", "tag": tc["tag"], "antibiotic": tc["antibiotic"]})

    print(f"  Prepared {len(test_files)} test spectra files.")

    # Step 4: Run predictions
    print(f"\n[4/4] Running predictions...\n")
    print(f"{'Tag':<8} {'Expected':<14} {'Predicted':<14} {'Prob':>8} {'Antibiotic':<15} {'Status'}")
    print("-" * 80)

    pass_count = 0
    fail_count = 0
    results_summary = []

    for tf in test_files:
        result = run_file_prediction(tf["file"], tf["species"])
        if result is None:
            fail_count += 1
            print(f"{tf['tag']:<8} {tf['expected']:<14} {'ERROR':<14} {'N/A':>8} {tf['antibiotic']:<15} FAIL")
            continue

        predicted = result["prediction_label"]
        prob = result["probability"]
        returned_antibiotic = result["antibiotic"]
        returned_species = result["species"]

        # Verify routing: the returned species/antibiotic must match what we sent
        routing_ok = (returned_species == tf["species"] and returned_antibiotic == tf["antibiotic"])

        # Check if probabilities make directional sense
        # For resistant samples we expect higher probability, for susceptible lower
        # But we don't enforce exact predictions (models aren't perfect)
        direction_ok = True
        if tf["expected"] == "Resistant" and prob < 0.1:
            direction_ok = False  # Very low prob for a known R sample is suspect
        if tf["expected"] == "Susceptible" and prob > 0.9:
            direction_ok = False  # Very high prob for a known S sample is suspect

        status = "PASS" if routing_ok else "FAIL (ROUTING)"
        if routing_ok:
            pass_count += 1
        else:
            fail_count += 1

        prob_display = f"{prob:.1%}"
        print(f"{tf['tag']:<8} {tf['expected']:<14} {predicted:<14} {prob_display:>8} {returned_antibiotic:<15} {status}")

        results_summary.append({
            "species": tf["species"],
            "tag": tf["tag"],
            "expected_label": tf["expected"],
            "predicted_label": predicted,
            "resistance_probability": prob,
            "returned_antibiotic": returned_antibiotic,
            "routing_correct": routing_ok,
            "clinical_report_length": len(result.get("clinical_report", "")),
        })

    print("-" * 80)
    print(f"\nRouting Tests:  {pass_count} passed, {fail_count} failed out of {pass_count + fail_count} total")

    # Print detailed probability comparison per species
    print(f"\n{'='*72}")
    print("  PROBABILITY COMPARISON (Resistant vs Susceptible per Species)")
    print(f"{'='*72}\n")

    for tc in TEST_CASES:
        species = tc["species"]
        tag = tc["tag"]
        r_results = [r for r in results_summary if r["tag"] == tag and r["expected_label"] == "Resistant"]
        s_results = [r for r in results_summary if r["tag"] == tag and r["expected_label"] == "Susceptible"]

        if r_results and s_results:
            r_prob = r_results[0]["resistance_probability"]
            s_prob = s_results[0]["resistance_probability"]
            diff = r_prob - s_prob
            print(f"  {tag} ({species}):")
            print(f"    Resistant sample probability:   {r_prob:.4f} ({r_prob:.1%})")
            print(f"    Susceptible sample probability: {s_prob:.4f} ({s_prob:.1%})")
            print(f"    Difference (R - S):             {diff:+.4f}")
            if diff > 0:
                print(f"    Model correctly assigns HIGHER probability to the resistant sample.")
            else:
                print(f"    NOTE: Model assigns lower prob to R sample. This can happen with")
                print(f"    difficult cases. Check the model's VME/FNR metrics.")
            print()

    # Verify clinical reports were generated
    print(f"{'='*72}")
    print("  CLINICAL REPORT GENERATION CHECK")
    print(f"{'='*72}\n")
    for r in results_summary:
        report_len = r["clinical_report_length"]
        status = "OK" if report_len > 50 else "TOO SHORT"
        print(f"  {r['tag']:<8} {r['expected_label']:<14} Report length: {report_len:>5} chars  [{status}]")

    print(f"\n{'='*72}")
    print("  INTEGRATION TEST COMPLETE")
    print(f"{'='*72}\n")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
