# NHAMCS ED 2015: Features vs Response for Data Science

Dataset: **ed2015-sas** (National Hospital Ambulatory Medical Care Survey – Emergency Department, 2015)  
~21k visits, 1,031 columns. Many columns use **-7 / -9** as missing/not ascertained.

---

## Response (target) columns – what to predict

Pick one or more depending on your question:

| Column      | Description                    | Use case                          |
|------------|--------------------------------|-----------------------------------|
| **WAITTIME** | Minutes until seen by provider | Regression: predict wait time     |
| **LOV**      | Length of visit (minutes)      | Regression: predict ED stay length |
| **LOS**      | Length of stay                 | Regression (often for admitted)    |
| **ADMIT** / **ADMITHOS** | Admitted to hospital (0/1) | **Classification: admission**     |
| **IMMEDR**   | Immediacy (triage 1–5)         | Classification: acuity             |
| **DIEDED**   | Died in ED                     | Rare-event classification          |
| **LEFTBTRI** / **LEFTATRI** / **LEFTAMA** | Left before/without seen, against medical advice | Classification |
| **BOARDED**  | Boarding time (minutes)        | Regression                         |
| **PAINSCALE**| Pain scale (0–10)              | Regression/classification          |
| **RETRNED**  | Return to ED within 72h        | Classification: revisit           |
| **NUMMED** / **TOTPROC** / **TOTDIAG** | # meds, procedures, diagnoses | Regression: intensity of care     |

**Disposition (mutually exclusive outcomes):**  
`RETRNED`, `TRANNH`, `TRANPSYC`, `TRANOTH`, `ADMITHOS`, `OBSHOS`, `OBSDIS`, `OTHDISP` — use one as a multiclass response or `ADMITHOS` as binary admission.

---

## Feature columns – predictors

Use these as **X** (known at or early in the visit). Drop design/weight columns from modeling.

### Demographics
- **AGE**, **AGER** (age group), **AGEDAYS** (if infant)
- **SEX**
- **RACER**, **RACERETH**, **ETHUN**, **ETHIM**
- **RESIDNCE** (residence type)

### Visit context (known at arrival)
- **VMONTH**, **VDAYR** (month, day of week)
- **ARRTIME** (arrival time)
- **ARREMS** (arrival by EMS), **AMBTRANSFER** (transfer from another facility)

### Payment / insurance
- **NOPAY**, **PAYPRIV**, **PAYMCARE**, **PAYMCAID**, **PAYWKCMP**, **PAYSELF**, **PAYNOCHG**, **PAYOTH**, **PAYDK**, **PAYTYPER**

### Triage / initial assessment
- **TEMPF**, **PULSE**, **RESPR**, **BPSYS**, **BPDIAS** (vitals)
- **POPCT** (pulse oximetry)
- **PAINSCALE** (if not your response)
- **SEEN72** (seen in ED in last 72 hours)

### Chief complaint / reason for visit
- **RFV1**–**RFV5** (reason for visit 1–5)
- **RFV13D**–**RFV53D** (3-digit RFV codes)
- **EPISODE** (initial vs follow-up)

### Injury / external cause
- **INJURY**, **INJR1**, **INJR2**, **INJPOISAD**, **INJURY72**, **INTENT15**
- **INJDETR**, **INJDETR1**, **INJDETR2**
- **CAUSE1**–**CAUSE3**, **CAUSE1R**–**CAUSE3R** (external cause codes)

### Comorbidities (from form)
- **ETOHAB**, **ALZHD**, **ASTHMA**, **CANCER**, **CEBVD**, **CKD**, **COPD**, **CHF**, **CAD**
- **DEPRN**, **DIABTYP1**, **DIABTYP2**, **ESRD**, **HPE**, **EDHIV**, **HYPLIPID**, **HTN**
- **OBESITY**, **OSA**, **OSTPRSIS**, **SUBSTAB**
- **NOCHRON**, **TOTCHRON** (number of chronic conditions)

### Diagnostics (can be features if “ordered” is known early)
- **CBC**, **BMP**, **CMP**, **EKG**, **XRAY**, **CATSCAN**, **MRI**, **ULTRASND**, **URINE**, etc.
- **ANYIMAGE**, **TOTDIAG**

### Procedures (often part of outcome; use as feature only if justified)
- **IVFLUIDS**, **SUTURE**, **CENTLINE**, **LUMBAR**, **NEBUTHER**, etc.
- **TOTPROC**

### Medications (count or indicators)
- **MED** (any med), **NUMGIV**, **NUMDIS**, **NUMMED**
- **GPMED1**–**GPMED30** (therapeutic class) if you need drug-level features

### Diagnoses (use recoded categories to avoid leakage)
- **DIAG1R**–**DIAG5R** (diagnosis groups); avoid raw **DIAG1**–**DIAG5** if they’re assigned after outcome.

### Facility / design (for stratification or fixed effects, not as main predictors)
- **REGION**, **MSA**, **HOSPCODE**
- **PATWT**, **EDWT**, **CSTRATM**, **CPSUM** — **use only for survey-weighted analysis; exclude from feature set for plain ML.**

---

## Exclude from features

- **PATWT**, **EDWT**, **CSTRATM**, **CPSUM** (survey weights/design — use only in weighted models).
- **PATCODE**, **HOSPCODE** (IDs — use for grouping, not as direct features).
- **YEAR**, **SETTYPE** (constant or design).
- Most **E*** (EHR) and **EOUT*** (discharge info) columns unless your question is about EHR or discharge process.
- Redundant encodings: prefer **AGER** over **AGE** if you want categories; prefer **DIAG*R** over raw **DIAG*** to control leakage.

---

## Quick start (Python)

```python
import pandas as pd
df = pd.read_sas("ed2015-sas.sas7bdat")

# Example: predict admission
response = "ADMITHOS"  # or "WAITTIME", "IMMEDR", etc.
features = [
    "AGE", "AGER", "SEX", "RACER", "ETHUN", "RESIDNCE",
    "VMONTH", "VDAYR", "ARREMS", "AMBTRANSFER", "PAYTYPER",
    "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "PAINSCALE",
    "RFV1", "RFV2", "EPISODE", "INJURY", "IMMEDR",
    "TOTCHRON", "ASTHMA", "HTN", "COPD", "CHF", "DIABTYP1", "DIABTYP2",
    "REGION",
]
# Drop rows where response is missing (e.g. -7, -9)
df_clean = df.replace(-7, np.nan).replace(-9, np.nan)
df_clean = df_clean.dropna(subset=[response])
X = df_clean[features].fillna(-999)  # or proper imputation
y = df_clean[response]
```

For **regression** (e.g. WAITTIME, LOV): use continuous `y`, handle negatives as missing.  
For **classification** (e.g. ADMITHOS): binarize `y` (e.g. 1 vs 0), then standard classification.
