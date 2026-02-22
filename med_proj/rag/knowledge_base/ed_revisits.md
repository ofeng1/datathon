# ED Revisits and Readmissions

## 72-Hour ED Revisits

A 72-hour ED revisit occurs when a patient returns to the emergency department within 72 hours of a prior ED discharge. This metric is widely used as a quality indicator, as high revisit rates may signal inadequate initial evaluation, premature discharge, or gaps in follow-up care.

### Risk Factors

- Chronic conditions (COPD, CHF, diabetes, chronic kidney disease)
- Mental health and substance use disorders
- Inadequate pain management
- Lack of primary care follow-up
- Social determinants: housing instability, transportation barriers
- Polypharmacy and medication non-adherence

### Mitigation Strategies

- Structured discharge planning with clear follow-up instructions
- Transitional care programs and nurse-led follow-up calls
- Medication reconciliation at discharge
- Care coordination with outpatient providers
- Social work referrals for patients with psychosocial needs

## Hospital Admissions from the ED

ED-to-inpatient admission is another key outcome. Predictive models can help with bed management, staffing, and early identification of patients likely to require admission.

### Common Predictors

- Triage acuity level (ESI)
- Vital sign abnormalities (tachycardia, hypotension, hypoxia, fever)
- Age and comorbidity burden
- Arrival by ambulance
- Prior ED visits and hospitalizations
- Chief complaint category

## NHAMCS ED Survey

The National Hospital Ambulatory Medical Care Survey (NHAMCS) Emergency Department component collects data on ED visits across the United States. Key variables include:

- **SEEN72**: Indicates whether the patient was seen in the same ED within the preceding 72 hours (1 = Yes, 2 = No).
- **ADMITHOS**: Indicates whether the patient was admitted to the hospital from the ED (1 = Yes, 0 = No).
- **IMMEDR**: Immediacy of visit (triage level).
- **LOV**: Length of visit in minutes.
- **PAINSCALE**: Pain severity rating.

These pre-computed flags allow direct use as outcome labels without requiring patient-level longitudinal linkage.
