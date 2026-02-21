from datetime import datetime, timezone

def parse_dt(s: str) -> datetime:
    if s is None or s == "":
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.fromisoformat(s.split(".")[0].replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0

def days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0