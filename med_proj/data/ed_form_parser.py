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


# Condition on admission / triage: Good=5, Fair=4, Stable=4, Guarded=3, Critical=1 or 2
_CONDITION_ESI = {
    "critical": 1.0,
    "guarded": 2.0,
    "stable": 3.0,
    "fair": 4.0,
    "good": 5.0,
}


def parse_ed_form_text(raw_text: str) -> Dict[str, Any]:
    """
    Parse raw text from an ED record form into a flat dict compatible with
    ChatEngine state (AGE, SEX, TEMPF, PULSE, RESPR, BPSYS, BPDIAS, POPCT,
    condition flags, etc.). Ignores uncorrelated fields (signatures, hospital #, etc.).
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    out: Dict[str, Any] = {}

    # Age: "Age: 45", "AGE: 45", "Age 45"
    m = re.search(r"\b(?:age|AGE)\s*:?\s*(\d{1,3})\b", text, re.I)
    if m:
        out["AGE"] = _float(m.group(1))

    # Sex: "Sex: M", "F", "Male", "Female"
    if re.search(r"\b(?:sex|Sex)\s*:?\s*[M1]\b|\b[M]\s*(?:\/|$)|Male\b", text):
        out["SEX"] = 1.0
    elif re.search(r"\b(?:sex|Sex)\s*:?\s*[F2]\b|\b[F]\s*(?:\/|$)|Female\b", text):
        out["SEX"] = 2.0

    # Temp: "TEMP 98.6", "TEMP: 98.6", "Temperature 101"
    m = re.search(r"\b(?:temp|TEMP|temperature)\s*:?\s*([\d.]+)", text, re.I)
    if m:
        v = _float(m.group(1))
        if v and 90 < v < 110:
            out["TEMPF"] = v

    # Pulse
    m = re.search(r"\b(?:pulse|PULSE)\s*:?\s*(\d+)", text, re.I)
    if m:
        out["PULSE"] = _float(m.group(1))

    # Resp
    m = re.search(r"\b(?:resp|RESP|respiration)\s*:?\s*(\d+)", text, re.I)
    if m:
        out["RESPR"] = _float(m.group(1))

    # B/P or Blood pressure: "120/80", "B/P 120/80"
    m = re.search(r"\b(?:B/P|BP|blood\s*pressure)\s*:?\s*(\d+)\s*/\s*(\d+)", text, re.I)
    if m:
        out["BPSYS"] = _float(m.group(1))
        out["BPDIAS"] = _float(m.group(2))

    # Pulse ox / SpO2
    m = re.search(r"\b(?:pulse\s*ox|PULSE\s*OX|spo2|SpO2|oxygen)\s*:?\s*(\d+)", text, re.I)
    if m:
        out["POPCT"] = _float(m.group(1))

    # Condition on admission -> triage
    for label, esi in _CONDITION_ESI.items():
        if re.search(r"\bcondition\s+on\s+admission\s*:?\s*" + label, text, re.I):
            out["IMMEDR"] = esi
            break
    if not out.get("IMMEDR") and re.search(r"\b(critical|guarded|stable|fair|good)\b", text, re.I):
        for label, esi in _CONDITION_ESI.items():
            if re.search(r"\b" + label + r"\b", text, re.I):
                out["IMMEDR"] = esi
                break

    # Significant medical history block -> run condition extractors
    m = re.search(
        r"SIGNIFICANT\s+MEDICAL\s+HISTORY\s*[\s:]*([\s\S]*?)(?=CURRENT\s+PRESCRIPTION|PROBLEM\s+ORIENTED|PHYSICAL\s+FINDINGS|LAB\s+&\s+X|$)",
        text,
        re.I,
    )
    if m:
        history_text = m.group(1).strip()
        if len(history_text) > 10:
            extracted = extract_all(history_text)
            for k, v in extracted.items():
                if k not in out or out[k] is None:
                    out[k] = v

    # Current prescription / meds block - can contain condition hints
    m = re.search(
        r"CURRENT\s+PRESCRIPTION\s+MEDICATION\s*[\s:]*([\s\S]*?)(?=SIGNIFICANT\s+MEDICAL|PROBLEM\s+ORIENTED|PHYSICAL\s+FINDINGS|USED\s+ANY|$)",
        text,
        re.I,
    )
    if m:
        med_text = m.group(1).strip()
        if len(med_text) > 5:
            med_extracted = extract_all(med_text)
            for k, v in med_extracted.items():
                if k not in out or out[k] is None:
                    out[k] = v

    # Disposition: Admitted -> high risk; we don't have a single field for "admitted" in state but could set a note. Skip for now.
    # Pain scale if present
    m = re.search(r"\b(?:pain|PAIN)\s*(?:scale)?\s*:?\s*(\d+)\s*(?:/\s*10)?", text, re.I)
    if m:
        out["PAINSCALE"] = _float(m.group(1))

    # Substance use in past 72 hrs
    if re.search(r"street\s*drugs\s*[Y\s]*Y|alcohol\s*[Y\s]*Y|used\s+any.*yes", text, re.I):
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
