#!/usr/bin/env python3
"""
Materialize REAL, label-verified DRIAMS-A demo spectra
======================================================

Copies genuine DRIAMS-A binned_6000 spectra with *ground-truth* resistance
labels (verified against the year's *_strat.csv metadata) into this directory,
so the Streamlit upload demo and manual testing exercise the model on real
data — not synthetic peaks.

Every file is named ``{Species}_{Antibiotic}_{LABEL}_{code8}.txt`` and its
provenance (sample code, year, antibiotic, true label) is recorded in
README.md. There is deliberately NO "borderline" file: DRIAMS AST labels are
only S / I / R, and I (intermediate) is dropped during training.

Run:
    .venv/bin/python test_spectra/prepare_real_spectra.py
"""

import os
import shutil

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIAMS_BINNED = os.path.join(PROJECT_ROOT, "DRIAMS-A", "binned_6000")
DRIAMS_ID = os.path.join(PROJECT_ROOT, "DRIAMS-A", "id")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Real DRIAMS-A isolates with known labels (same codes used by integration_test.py)
CASES = [
    ("Staphylococcus aureus", "Oxacillin",
     ("0057f21c-94f9-4f35-b8ff-969e187ef1f8", "2016"),
     ("000a2090-67fb-4556-8d34-3b758508be98", "2016")),
    ("Klebsiella pneumoniae", "Meropenem",
     ("94945229-059b-47a9-ab4f-33560ac3383b", "2015"),
     ("031185e1-74a9-4f00-aea0-70d4add3da21", "2015")),
    ("Escherichia coli", "Ceftriaxone",
     ("006fd66b-55ce-42cd-9e4d-e404f7886ec5", "2016"),
     ("002c4fc6-42a6-4cd5-8ee8-207038bea4f9", "2016")),
]


def true_label(year: str, code: str, antibiotic: str) -> str:
    """Look up the ground-truth AST label for a sample from *_strat.csv."""
    df = pd.read_csv(os.path.join(DRIAMS_ID, year, f"{year}_strat.csv"), low_memory=False)
    row = df[df["code"] == code]
    if row.empty:
        raise ValueError(f"code {code} not found in {year}_strat.csv")
    return str(row.iloc[0].get(antibiotic))


def main():
    manifest = ["# Real DRIAMS-A Demo Spectra\n",
                "These are genuine clinical isolates with ground-truth AST labels ",
                "verified against DRIAMS-A `*_strat.csv` metadata. Upload them in the ",
                "Streamlit app to exercise the models on real data.\n",
                "\n| File | Species | Antibiotic | True label | DRIAMS code | Year |",
                "\n|------|---------|-----------|-----------|-------------|------|\n"]

    for species, antibiotic, r_case, s_case in CASES:
        for (code, year), expected in [(r_case, "R"), (s_case, "S")]:
            label = true_label(year, code, antibiotic)
            assert label == expected, (
                f"Label mismatch for {code}: metadata says {label!r}, expected {expected!r}"
            )
            label_word = {"R": "RESISTANT", "S": "SUSCEPTIBLE"}[label]
            src = os.path.join(DRIAMS_BINNED, year, f"{code}.txt")
            dest_name = f"{species.replace(' ', '_')}_{antibiotic}_{label_word}_{code[:8]}.txt"
            dest = os.path.join(OUTPUT_DIR, dest_name)
            shutil.copy2(src, dest)
            print(f"  wrote {dest_name}  (true label: {label})")
            manifest.append(
                f"| `{dest_name}` | {species} | {antibiotic} | **{label_word}** | `{code}` | {year} |\n"
            )

    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.writelines(manifest)
    print("  wrote README.md manifest")


if __name__ == "__main__":
    main()
