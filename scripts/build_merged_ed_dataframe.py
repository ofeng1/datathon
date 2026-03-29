#!/usr/bin/env python3
"""
Build a merged NHAMCS ED dataframe by loading multiple year ZIPs and
concatenating them into one pandas DataFrame.

This is the "join ZIPs onto one dataframe" logic currently shown in a notebook.
The output is saved under `data/processed/` for downstream modeling/analysis.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Ensure `src/` is importable when running from repo root:
#   ./.venv/bin/python scripts/build_merged_ed_dataframe.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from common.io import ensure_dir  # noqa: E402


DEFAULT_ZIP_PATHS = [
    "ed2015-sas.sas7bdat.zip",
    "ed2016_sas.zip",
    "ed2017_sas.zip",
    "ed2018_sas.zip",
    "ed2019_sas.zip",
    "ed2020_sas.zip",
    "ed2021_sas.zip",
]


def _clear_interim_dir() -> None:
    """
    `DataLoader.load_data()` extracts ZIP contents into `data/interim/` and does
    not remove older extracted SAS files. Clearing prevents cross-year mixing.
    """

    interim_dir = REPO_ROOT / "data" / "interim"
    if not interim_dir.exists():
        return

    for child in interim_dir.iterdir():
        if child.is_file():
            child.unlink()

def _load_zip_with_pyreadstat(zip_filename: str) -> pd.DataFrame:
    """
    Load the largest extracted `.sas7bdat` from `data/raw/<zip_filename>` using
    `pyreadstat` (more stable than `pandas.read_sas` in this environment).
    """
    import pyreadstat  # imported lazily so script still runs without it

    raw_dir = REPO_ROOT / "data" / "raw"
    interim_dir = REPO_ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / zip_filename
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=interim_dir)

    sas_files = list(interim_dir.glob("*.sas7bdat"))
    if not sas_files:
        raise FileNotFoundError("No .sas7bdat found after extraction.")

    sas_path = max(sas_files, key=lambda p: p.stat().st_size)
    df, _meta = pyreadstat.read_sas7bdat(str(sas_path))
    return df


def build_merged_dataframe(zip_paths: list[str], clear_interim: bool = True) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    # Prefer pyreadstat if available (installed into local venv).
    try:
        import pyreadstat  # noqa: F401
        use_pyreadstat = True
    except ModuleNotFoundError:
        use_pyreadstat = False

    # Fallback to the existing DataLoader if pyreadstat isn't available.
    if not use_pyreadstat:
        from data.data_loader import DataLoader  # noqa: E402

        loader = DataLoader()

        for zip_filename in zip_paths:
            if clear_interim:
                _clear_interim_dir()
            dfs.append(loader.load_data(zip_filename))
    else:
        for zip_filename in zip_paths:
            if clear_interim:
                _clear_interim_dir()
            dfs.append(_load_zip_with_pyreadstat(zip_filename))

    if not dfs:
        raise ValueError("No dataframes were loaded; check zip_paths.")

    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/processed/merged_ed_dataframe.pkl",
        help="Output pickle path under the repo (default: data/processed/merged_ed_dataframe.pkl).",
    )
    parser.add_argument(
        "--no-clear-interim",
        action="store_true",
        help="Disable clearing data/interim/ between ZIP loads (not recommended).",
    )
    args = parser.parse_args()

    out_path = (REPO_ROOT / args.output).resolve()
    ensure_dir(str(out_path.parent))

    df = build_merged_dataframe(DEFAULT_ZIP_PATHS, clear_interim=not args.no_clear_interim)
    df.to_pickle(out_path)
    print(f"Saved merged dataframe: {out_path}")
    print(f"Merged shape: {df.shape}")


if __name__ == "__main__":
    main()

