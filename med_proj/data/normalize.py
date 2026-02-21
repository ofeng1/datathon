from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from med_proj.common.logging import get_loggerlog


@dataclass
class MappingConfig:
    patient_id_col: str
    encounter_id_col: str

    start_time_col: Optional[str] = None
    end_time_col: Optional[str] = None

    start_date_col: Optional[str] = None
    start_clock_col: Optional[str] = None
    end_date_col: Optional[str] = None
    end_clock_col: Optional[str] = None

    admitted_col: Optional[str] = None
    admitted_positive_values: Optional[List[Any]] = None


def _require_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns (sample): {list(df.columns)[:50]}")


def _to_iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_clock(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            h = int(parts[0])
            m = int(parts[1])
            return h * 60 + m
        except Exception:
            return None
    # digits form
    try:
        digits = int(float(s))
    except Exception:
        return None
    # HHMM or HHMMSS
    if digits <= 2359:
        h = digits // 100
        m = digits % 100
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
        return None
    # HHMMSS
    if digits <= 235959:
        h = digits // 10000
        m = (digits // 100) % 100
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
    return None


def _combine_date_clock(date_val, clock_val) -> Optional[datetime]:
    if date_val is None or (isinstance(date_val, float) and np.isnan(date_val)):
        return None

    # Pandas may already parse SAS date into datetime-like
    if isinstance(date_val, datetime):
        base_date = date_val.date()
    else:
        try:
            base_date = pd.to_datetime(date_val).date()
        except Exception:
            return None

    mins = _parse_clock(clock_val)
    if mins is None:
        # default midnight if clock missing
        mins = 0
    h = mins // 60
    m = mins % 60
    return datetime(base_date.year, base_date.month, base_date.day, h, m, tzinfo=timezone.utc)


def normalize_sas_to_encounters(df_raw: pd.DataFrame, cfg: MappingConfig) -> pd.DataFrame:
    df = df_raw.copy()

    _require_cols(df, [cfg.patient_id_col, cfg.encounter_id_col])

    # Determine how timestamps are provided
    has_direct_ts = cfg.start_time_col is not None and cfg.start_time_col in df.columns
    has_split_ts = cfg.start_date_col is not None and cfg.start_date_col in df.columns

    if not has_direct_ts and not has_split_ts:
        raise ValueError(
            "No valid timestamp mapping found.\n"
            "Provide either start_time_col OR (start_date_col + start_clock_col)."
        )

    # Build start/end datetimes
    if has_direct_ts:
        _require_cols(df, [cfg.start_time_col])
        start_dt = pd.to_datetime(df[cfg.start_time_col], errors="coerce", utc=True)
        if cfg.end_time_col and cfg.end_time_col in df.columns:
            end_dt = pd.to_datetime(df[cfg.end_time_col], errors="coerce", utc=True)
        else:
            end_dt = start_dt
    else:
        _require_cols(df, [cfg.start_date_col])
        start_dt = df.apply(lambda r: _combine_date_clock(r[cfg.start_date_col], r.get(cfg.start_clock_col)), axis=1)
        start_dt = pd.to_datetime(start_dt, errors="coerce", utc=True)

        if cfg.end_date_col and cfg.end_date_col in df.columns:
            end_dt = df.apply(lambda r: _combine_date_clock(r[cfg.end_date_col], r.get(cfg.end_clock_col)), axis=1)
            end_dt = pd.to_datetime(end_dt, errors="coerce", utc=True)
        else:
            end_dt = start_dt

    out = pd.DataFrame({
        "patient_id": df[cfg.patient_id_col].astype(str),
        "encounter_id": df[cfg.encounter_id_col].astype(str),
        "start_time": start_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": end_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "encounter_type": "ed",
    })

    # Optional admitted flag
    if cfg.admitted_col and cfg.admitted_col in df.columns:
        pos = set(cfg.admitted_positive_values or [1, "1", "Y", "YES", True, "True"])
        out["admitted"] = df[cfg.admitted_col].apply(lambda x: x in pos)

    # Drop bad timestamps
    out = out.dropna(subset=["start_time"])
    log.info("Normalized to encounters: rows=%d", len(out))
    return out