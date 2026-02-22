"""Regex-based extractors that pull structured clinical values from free text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Each extractor returns a dict of field->value pairs (may be empty).

_WEEKDAYS = {
    "sunday": 1, "sun": 1,
    "monday": 2, "mon": 2,
    "tuesday": 3, "tue": 3, "tues": 3,
    "wednesday": 4, "wed": 4,
    "thursday": 5, "thu": 5, "thurs": 5,
    "friday": 6, "fri": 6,
    "saturday": 7, "sat": 7,
}

_CONDITION_TO_NHAMCS = {
    r"copd|chronic\s*obstructive":                          "COPD",
    r"chf|congestive\s*heart\s*failure|heart\s*failure":    "CHF",
    r"cad|coronary\s*artery":                               "CAD",
    r"asthma":                                              "ASTHMA",
    r"ckd|chronic\s*kidney|kidney\s*disease":               "CKD",
    r"esrd|end[\s-]*stage\s*renal":                         "ESRD",
    r"hypertension|htn|\bhigh\s*blood\s*pressure":          "HTN",
    r"diabet(?:es|ic)\s*(?:type\s*)?(?:1|i(?!i))\b":       "DIABTYP1",
    r"diabet(?:es|ic)\s*(?:type\s*)?(?:2|ii)\b":           "DIABTYP2",
    r"diabet(?:es|ic)":                                     "DIABTYP0",
    r"cancer|malignan|lymphoma|leukemia|tumor|oncol":       "CANCER",
    r"depression|depressed|major\s*depress":                "DEPRN",
    r"cerebrovascular|stroke|\bcva\b":                      "CEBVD",
    r"alzheimer|dementia":                                  "ALZHD",
    r"hyperlipid|high\s*cholesterol":                       "HYPLIPID",
    r"obesity|obese|\bbmi\s*>\s*3[0-9]":                    "OBESITY",
    r"sleep\s*apnea|\bosa\b":                               "OSA",
    r"osteoporosis":                                        "OSTPRSIS",
    r"\bhiv\b|human\s*immunodeficiency":                    "EDHIV",
    r"alcohol(?:ism|ic|\s*abuse|use\s*disorder)":           "ETOHAB",
    r"substance\s*abuse|drug\s*abuse|sud\b":                "SUBSTAB",
}


def _float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def extract_age(text: str) -> Dict[str, Any]:
    m = re.search(r'\b(\d{1,3})\s*[-]?\s*(yr|year|yo|y/?o)\s*(old)?', text, re.I)
    if m:
        return {"AGE": float(m.group(1))}
    m = re.search(r'\bage\s*(?:[:=]|is|to)?\s*(\d{1,3})\b', text, re.I)
    if m:
        return {"AGE": float(m.group(1))}
    return {}


def extract_sex(text: str) -> Dict[str, Any]:
    if re.search(r'\b(female|woman|girl)\b', text, re.I):
        return {"SEX": 2.0}
    if re.search(r'\b(male|man|boy)\b', text, re.I):
        return {"SEX": 1.0}
    m = re.search(r'\b(sex|gender)\s*[:=]?\s*([MFmf])\b', text)
    if m:
        return {"SEX": 1.0 if m.group(2).upper() == "M" else 2.0}
    return {}


def extract_vitals(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    m = re.search(r'\btemp(?:erature|f)?\s*(?:[:=]|is|of)?\s*([\d.]+)', text, re.I)
    if m:
        v = _float(m.group(1))
        if v and v > 50:
            out["TEMPF"] = v

    m = re.search(r'\b(?:pulse|hr|heart\s*rate)\s*(?:[:=]|is|of)?\s*(\d+)', text, re.I)
    if m:
        out["PULSE"] = _float(m.group(1))

    m = re.search(r'\b(?:resp|rr|respiratory\s*rate)\s*(?:[:=]|is|of)?\s*(\d+)', text, re.I)
    if m:
        out["RESPR"] = _float(m.group(1))

    m = re.search(r'\b(?:bp|blood\s*pressure)\s*[:=]?\s*(\d+)\s*/\s*(\d+)', text, re.I)
    if m:
        out["BPSYS"] = _float(m.group(1))
        out["BPDIAS"] = _float(m.group(2))

    m = re.search(r'\b(?:spo2|o2\s*sat|oxygen|sat)\s*[:=]?\s*(\d+)', text, re.I)
    if m:
        out["POPCT"] = _float(m.group(1))

    return out


def extract_pain(text: str) -> Dict[str, Any]:
    m = re.search(r'\bpain\s*(?:scale|score)?\s*(?:[:=]|is|of)?\s*(\d+)\s*(?:/\s*10)?', text, re.I)
    if m:
        return {"PAINSCALE": _float(m.group(1))}
    return {}


def extract_arrtime(text: str) -> Dict[str, Any]:
    m = re.search(r'\b(?:arriv\w*|arrival)\s*(?:time|at)?\s*[:=]?\s*(\d{1,2}):(\d{2})\s*(am|pm)?', text, re.I)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if m.group(3):
            ampm = m.group(3).lower()
            if ampm == "pm" and h < 12:
                h += 12
            elif ampm == "am" and h == 12:
                h = 0
        return {"ARRTIME": float(h * 100 + mi)}

    m = re.search(r'\b(?:arriv\w*|arrival)\s*(?:time)?\s*[:=]?\s*(\d{3,4})\b', text, re.I)
    if m:
        return {"ARRTIME": _float(m.group(1))}
    return {}


def extract_lov(text: str) -> Dict[str, Any]:
    m = re.search(r'\b(?:lov|length of visit|been here|here for)\s*[:=]?\s*([\d.]+)\s*(hr|hour|h|min|minute|m)\w*', text, re.I)
    if m:
        val = _float(m.group(1))
        unit = m.group(2).lower()
        if val is not None:
            if unit.startswith("h"):
                return {"LOV": val * 60}
            return {"LOV": val}
    m = re.search(r'\b(\d+)\s*(hr|hour|h)\s*(visit|in the ed|in ed)', text, re.I)
    if m:
        val = _float(m.group(1))
        if val is not None:
            return {"LOV": val * 60}
    return {}


def extract_chronic(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    matched_flags: set = set()

    for pattern, col in _CONDITION_TO_NHAMCS.items():
        if col in matched_flags:
            continue
        if re.search(pattern, text, re.I):
            out[col] = 1.0
            matched_flags.add(col)

    if "DIABTYP0" in out and ("DIABTYP1" in out or "DIABTYP2" in out):
        del out["DIABTYP0"]

    explicit = re.search(r'(\d+)\s*(?:chronic)?\s*(?:condition|comorbidit|disease)', text, re.I)
    if explicit:
        out["TOTCHRON"] = _float(explicit.group(1))
    elif matched_flags:
        out["TOTCHRON"] = float(len(matched_flags))

    if matched_flags:
        out["NOCHRON"] = 0.0
    elif "TOTCHRON" in out and out["TOTCHRON"] == 0:
        out["NOCHRON"] = 1.0

    return out


def extract_injury(text: str) -> Dict[str, Any]:
    if re.search(r'\b(injury|injured|trauma|fall|accident|laceration|fracture)\b', text, re.I):
        return {"INJURY": 1.0}
    return {}


def extract_substance(text: str) -> Dict[str, Any]:
    if re.search(r'\b(substance\s*abuse|drug\s*abuse|alcoholi|intoxicat|overdose|drug\s*use)\b', text, re.I):
        return {"SUBSTAB": 1.0}
    return {}


def extract_day_of_week(text: str) -> Dict[str, Any]:
    for name, val in _WEEKDAYS.items():
        if re.search(r'\b' + name + r'\b', text, re.I):
            return {"VDAYR": float(val)}
    return {}


def extract_triage(text: str) -> Dict[str, Any]:
    m = re.search(r'\b(?:triage|esi|acuity|immedr)\s*[:=]?\s*(\d)\b', text, re.I)
    if m:
        return {"IMMEDR": _float(m.group(1))}
    return {}


def extract_prior_visits(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(r'(\d+)\s*(?:prior|previous)\s*(?:ed)?\s*visit', text, re.I)
    if m:
        out["prior_ed_30d"] = int(m.group(1))
    m = re.search(r'\b(?:last visit|last ed|days since)\s*[:=]?\s*(\d+)\s*(?:day)?', text, re.I)
    if m:
        out["days_since_last_encounter"] = float(m.group(1))
    return out


_ALL_EXTRACTORS = [
    extract_age,
    extract_sex,
    extract_vitals,
    extract_pain,
    extract_arrtime,
    extract_lov,
    extract_chronic,
    extract_injury,
    extract_substance,
    extract_day_of_week,
    extract_triage,
    extract_prior_visits,
]


def extract_all(text: str) -> Dict[str, Any]:
    """Run every extractor and merge results into a single dict."""
    merged: Dict[str, Any] = {}
    for fn in _ALL_EXTRACTORS:
        merged.update(fn(text))
    return merged
