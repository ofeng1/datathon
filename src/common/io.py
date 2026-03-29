from datetime import datetime, timezone
from pathlib import Path

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_dt(s):
    if s is None:
        return None
    if isinstance(s, datetime):
        dt = s
    else:
        ss = str(s).strip()
        if ss == "" or ss.lower() == "nan":
            return None
        # Accept ISO; if no timezone, assume UTC
        try:
            dt = datetime.fromisoformat(ss.replace("Z", "+00:00"))
        except ValueError:
            # try removing subseconds
            dt = datetime.fromisoformat(ss.split(".")[0].replace("Z", "+00:00"))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0

def days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0