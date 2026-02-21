from pathlib import Path
import os

def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)