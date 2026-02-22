"""Parse text extracted from an ED / Emergency Dept Record form into state for the chatbot."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from med_proj.chatbot.extractors import extract_all


def _float(s: str) -> Optional[float]:
    if not s or not s.strip():
        return None
    try:
        return float(re.sub(r"[^\d.\-]", "", s.strip()) or 0)
    except (ValueError, TypeError):
        return None


def _int(s: str) -> Optional[int]:
    v = _float(s)
    return int(v) if v is not None else None


# Triage/ESI: word form -> numeric
_CONDITION_ESI = {
    "critical": 1.0,
    "immediate": 1.0,
    "emergent": 2.0,
    "guarded": 2.0,
    "urgent": 3.0,
    "stable": 3.0,
    "semi-urgent": 4.0,
    "semiurgent": 4.0,
    "fair": 4.0,
    "non-urgent": 5.0,
    "nonurgent": 5.0,
    "good": 5.0,
}


def _normalize_form_text(raw: str) -> str:
    """Merge label/value on consecutive lines so 'Age\\n45' becomes 'Age: 45' for regex matching."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Common form labels that often appear above or left of a value on next line/cell
    labels = [
        "Age", "Sex", "Gender", "DOB", "Date of Birth",
        "Temp", "Temperature", "TEMP", "Pulse", "HR", "Heart Rate", "PULSE",
        "Resp", "RR", "Respiration", "RESP", "BP", "B/P", "Blood Pressure",
        "SpO2", "O2 sat", "Pulse ox", "Oxygen", "POPCT",
        "Pain", "Pain scale", "PAIN", "ESI", "Triage", "Acuity",
        "Chief complaint", "Chief Complaint", "CC", "Complaint",
        "Allergies", "Allergy", "Medications", "Meds", "Current medications",
        "Disposition", "Discharge", "Diagnosis", "DX",
        "Condition on admission", "Condition",
    ]
    lines = text.split("\n")
    out_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # If this line is only a label (no digits, short), next line might be the value
        if i + 1 < len(lines) and stripped:
            next_stripped = lines[i + 1].strip()
            # Label-like: ends with : or is one of our labels and next line looks like a value
            for lbl in labels:
                if re.match(r"^" + re.escape(lbl) + r"\s*:?\s*$", stripped, re.I):
                    if next_stripped and (re.search(r"[\d/.]", next_stripped) or len(next_stripped) < 80):
                        out_lines.append(stripped.rstrip(":") + ": " + next_stripped)
                        lines[i + 1] = ""  # consume next
                        break
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def parse_ed_form_text(raw_text: str) -> Dict[str, Any]:
    """
    Parse raw text from an ED record form into a flat dict compatible with
    ChatEngine state. Extracts vitals, ESI/triage, conditions, and optional
    display fields (chief complaint, disposition, allergies, etc.).
    """
    text = _normalize_form_text(raw_text)
    # Also keep a single-line version for patterns that span lines poorly
    text_flat = re.sub(r"\s+", " ", text)
    out: Dict[str, Any] = {}

    # ---- ESI / Triage (1–5) ----
    m = re.search(r"\b(?:ESI|Triage\s*(?:acuity)?|Acuity)\s*:?\s*(\d)\b", text, re.I)
    if m:
        out["IMMEDR"] = _float(m.group(1))
    if out.get("IMMEDR") is None:
        for label, esi in _CONDITION_ESI.items():
            if re.search(r"\b" + re.escape(label) + r"\b", text, re.I):
                out["IMMEDR"] = esi
                break
    if out.get("IMMEDR") is None and re.search(r"\bcondition\s+on\s+admission\s*:?\s*", text, re.I):
        for label, esi in _CONDITION_ESI.items():
            if re.search(r"condition\s+on\s+admission\s*:?\s*" + label, text, re.I):
                out["IMMEDR"] = esi
                break

    # ---- Age ----
    m = re.search(r"\b(?:age|AGE)\s*:?\s*(\d{1,3})\b", text, re.I)
    if m:
        out["AGE"] = _float(m.group(1))
    if out.get("AGE") is None:
        m = re.search(r"\b(\d{1,3})\s*(?:years?\s*old|yo|y\.o\.)\b", text, re.I)
        if m:
            out["AGE"] = _float(m.group(1))

    # ---- Sex / Gender ----
    if re.search(r"\b(?:sex|gender)\s*:?\s*[M1]\b|\bM\s*(?:\/|$|\s)|Male\b", text, re.I):
        out["SEX"] = 1.0
    elif re.search(r"\b(?:sex|gender)\s*:?\s*[F2]\b|\bF\s*(?:\/|$|\s)|Female\b", text, re.I):
        out["SEX"] = 2.0

    # ---- Temp ----
    m = re.search(r"\b(?:temp|TEMP|temperature)\s*:?\s*([\d.]+)", text, re.I)
    if m:
        v = _float(m.group(1))
        if v and 90 < v < 110:
            out["TEMPF"] = v

    # ---- Pulse / HR ----
    m = re.search(r"\b(?:pulse|PULSE|HR|heart\s*rate)\s*:?\s*(\d+)", text, re.I)
    if m:
        out["PULSE"] = _float(m.group(1))

    # ---- Resp ----
    m = re.search(r"\b(?:resp|RESP|RR|respiration|respiratory\s*rate)\s*:?\s*(\d+)", text, re.I)
    if m:
        out["RESPR"] = _float(m.group(1))

    # ---- BP ----
    m = re.search(r"\b(?:B/P|BP|blood\s*pressure)\s*:?\s*(\d+)\s*/\s*(\d+)", text, re.I)
    if m:
        out["BPSYS"] = _float(m.group(1))
        out["BPDIAS"] = _float(m.group(2))
    if out.get("BPSYS") is None:
        m = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmHg|mm\s*hg)?", text_flat)
        if m:
            s, d = _float(m.group(1)), _float(m.group(2))
            if s and d and 60 < s < 250 and 30 < d < 150:
                out["BPSYS"], out["BPDIAS"] = s, d

    # ---- SpO2 / Pulse ox ----
    m = re.search(r"\b(?:pulse\s*ox|PULSE\s*OX|spo2|SpO2|O2\s*sat|oxygen\s*sat)\s*:?\s*(\d+)", text, re.I)
    if m:
        out["POPCT"] = _float(m.group(1))

    # ---- Pain scale ----
    m = re.search(r"\b(?:pain|PAIN)\s*(?:scale)?\s*:?\s*(\d+)\s*(?:/\s*10)?", text, re.I)
    if m:
        out["PAINSCALE"] = _float(m.group(1))

    # ---- Chief complaint (display) ----
    m = re.search(
        r"(?:chief\s*complaint|CC|complaint)\s*:?\s*([^\n]+)",
        text,
        re.I,
    )
    if m:
        cc = m.group(1).strip()
        if len(cc) > 2 and len(cc) < 500 and not re.match(r"^(?:History|Medication|Vital|Temp|Pulse|BP|ESI)\b", cc, re.I):
            out["chief_complaint"] = cc

    # ---- Allergies (display) ----
    m = re.search(
        r"(?:allerg(?:y|ies)|NKDA|NKA)\s*:?\s*([^\n]+)",
        text,
        re.I,
    )
    if m:
        alg = m.group(1).strip()
        if len(alg) > 1 and len(alg) < 400:
            out["allergies"] = alg
    if out.get("allergies") is None and re.search(r"\b(?:NKDA|NKA|no\s*known\s*drug\s*allerg)\b", text, re.I):
        out["allergies"] = "No known drug allergies"

    # ---- Disposition (display) ----
    m = re.search(
        r"disposition\s*:?\s*([^\n]+)",
        text,
        re.I,
    )
    if m:
        disp = m.group(1).strip()
        if len(disp) < 200:
            out["disposition"] = disp

    # ---- Diagnosis / DX (display) ----
    m = re.search(
        r"(?:diagnosis|DX|final\s*diagnosis)\s*:?\s*([^\n]+)",
        text,
        re.I,
    )
    if m:
        dx = m.group(1).strip()
        if len(dx) > 2 and len(dx) < 500:
            out["diagnosis_notes"] = dx

    # ---- Conditions: run extractors on full text and on specific sections ----
    for block_regex, name in [
        (
            r"SIGNIFICANT\s+MEDICAL\s+HISTORY\s*[\s:]*([\s\S]*?)(?=CURRENT\s+PRESCRIPTION|PROBLEM\s+ORIENTED|PHYSICAL\s+FINDINGS|LAB\s+&\s+X|$)",
            "history",
        ),
        (
            r"CURRENT\s+PRESCRIPTION\s+MEDICATION\s*[\s:]*([\s\S]*?)(?=SIGNIFICANT\s+MEDICAL|PROBLEM\s+ORIENTED|PHYSICAL\s+FINDINGS|USED\s+ANY|$)",
            "meds",
        ),
        (
            r"(?:HISTORY|Medical\s+History|PMH|Past\s+Medical)[\s:]*([\s\S]*?)(?=(?:Medication|Allerg|Vital|Temp|Pulse|BP|ESI|Triage|Disposition|$))",
            "history_alt",
        ),
    ]:
        m = re.search(block_regex, text, re.I)
        if m:
            block = m.group(1).strip()
            if len(block) > 10:
                extracted = extract_all(block)
                for k, v in extracted.items():
                    if k not in out or out[k] is None:
                        out[k] = v
    # Run on full text so we don't miss conditions mentioned elsewhere
    full_extracted = extract_all(text_flat)
    for k, v in full_extracted.items():
        if k not in out or out[k] is None:
            out[k] = v

    # Substance use
    if re.search(r"street\s*drugs\s*[Y\s]*Y|alcohol\s*[Y\s]*Y|used\s+any.*yes|substance\s*use\s*:\s*yes", text, re.I):
        out["SUBSTAB"] = 1.0

    return out


def pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF file."""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)
