from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable, Optional, Union
import zipfile

import pandas as pd


def _default_candidates() -> list[Path]:
    data_dir = Path(__file__).resolve().parent
    repo_root = data_dir.parent
    names = ["ed2015-sas.sas7bdat", "ed2015-sas.sas7bdat.zip"]
    return [data_dir / n for n in names] + [repo_root / n for n in names]


def _decode_bytes_object_columns(df: pd.DataFrame, *, encoding: str) -> pd.DataFrame:
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return df

    out = df.copy()
    for c in obj_cols:
        s = out[c]
        if not s.map(lambda v: isinstance(v, (bytes, bytearray))).any():
            continue
        out[c] = s.map(
            lambda v: v.decode(encoding, errors="replace")
            if isinstance(v, (bytes, bytearray))
            else v
        )
    return out


def load_sas_data(
    path: Optional[Union[str, Path]] = None,
    *,
    encodings: Iterable[str] = ("cp1252", "latin-1", "utf-8"),
) -> pd.DataFrame:
    """
    Load the ED2015 SAS dataset from either:
    - a `.sas7bdat` file, or
    - a `.zip` containing a `.sas7bdat`

    Handles the common UTF-8 decode error (byte 0xa0) by trying `cp1252`
    (Windows) and then `latin-1`.
    """
    if path is None:
        candidates = _default_candidates()
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "No SAS file found. Expected one of:\n"
                + "\n".join(f"- {p}" for p in candidates)
            )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    enc_list = list(encodings)

    def _read_sas(sas_path: Path) -> pd.DataFrame:
        last_err: Exception | None = None
        for enc in enc_list:
            try:
                df = pd.read_sas(sas_path, format="sas7bdat", encoding=enc)
                return _decode_bytes_object_columns(df, encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
        raise RuntimeError(
            f"Failed to decode {sas_path} with encodings {enc_list}. Last error: {last_err}"
        )

    if path.suffix.lower() == ".sas7bdat":
        return _read_sas(path)

    if path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            with zipfile.ZipFile(path) as z:
                members = [n for n in z.namelist() if n.lower().endswith(".sas7bdat")]
                if not members:
                    raise ValueError(f"No .sas7bdat found inside zip: {path}")
                extracted = z.extract(members[0], path=td_path)
                sas_path = Path(extracted)
            return _read_sas(sas_path)

    raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    df = load_sas_data()
    print(df.shape)
    print(df.head().to_string())

