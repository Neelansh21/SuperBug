"""
Clinical Prompt Templates
=========================

Defines the prompt template used to guide the Gemini LLM to generate
structured, clinically safe, and actionable AMR interpretation reports.

Report structure enforced:
1. Executive Summary
2. Machine Learning Diagnostic Insights
3. Therapeutic Actionability
4. Biosafety & Infection Control protocols
"""

from langchain_core.prompts import PromptTemplate

CLINICAL_REPORT_PROMPT = PromptTemplate.from_template(
    """You are an expert Clinical Microbiologist and Infectious Disease Specialist.
Generate a structured Clinical Interpretation and Actionability Report based on the following diagnostic results, patient metadata, and clinical reference guidelines.

[DIAGNOSTIC RESULT]
Pathogen Species: {species}
Tested Antibiotic: {antibiotic}
Machine Learning Prediction: {prediction_label} ({probability:.1f}% probability of resistance)

[PATIENT CLINICAL METADATA]
Clinical Workstation: {workstation}
Acquisition Date: {acquisition_date}
Collection Year: {year}

[CLINICAL GUIDELINES & RECOMMENDATIONS]
{guidelines}

Generate the report in markdown format. Use the following structure:
# CLINICAL INTERPRETATION REPORT: {species} ({clinical_name})

## 1. Executive Summary
Provide a 2-3 sentence clinical summary of the finding. Explicitly state whether the strain is predicted to be Resistant or Susceptible to the tested antibiotic, and the level of confidence.

## 2. Machine Learning Diagnostic Insight
Discuss the diagnostic model's prediction:
- State the prediction probability: {probability:.1f}% probability of resistance.
- Discuss the clinical implication of this prediction (e.g., for S. aureus and Oxacillin, this represents Methicillin-Resistant Staphylococcus aureus / MRSA; for E. coli and Ceftriaxone, this suggests ESBL production; for K. pneumoniae and Meropenem, this indicates carbapenem resistance).
- State that this result is generated from MALDI-TOF mass spectrometry profile analysis via machine learning.

## 3. Therapeutic Actionability
Provide actionable treatment advice based on the provided [CLINICAL GUIDELINES & RECOMMENDATIONS]:
- **Recommended Alternative Therapies**: List the recommended antibiotics. Discuss any patient context (e.g., daptomycin constraint for pneumonia, vancomycin troughs, linezolid duration warnings, piperacillin-tazobactam bacteremia warning).
- **Therapies to Avoid**: List the classes of antibiotics or specific drugs that must be avoided.

## 4. Infection Control & Biosafety
List the isolation precautions and control measures required for this patient (e.g., contact isolation, rectal swab screening, notifying infection control, chlorhexidine bathing).
"""
)
