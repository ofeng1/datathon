"""
Go through each column of the SAS dataframe and store it under output_columns/.
One file per column (CSV) plus a manifest JSON with names and dtypes.
"""
import json
import re
from pathlib import Path

import pandas as pd


def sanitize_filename(name: str) -> str:
    """Make a safe filename from a column name."""
    safe = re.sub(r"[^\w\-.]", "_", str(name))
    return safe or "unnamed"


def store_columns(
    df: pd.DataFrame,
    out_dir: str | Path = "output_columns",
    format: str = "csv",
) -> Path:
    """
    Write each column to a file under out_dir and save a manifest.

    Parameters
    ----------
    df : DataFrame (e.g. from pd.read_sas)
    out_dir : directory for column files
    format : "csv" or "parquet" (per-column parquet is one column per file)

    Returns
    -------
    Path to the manifest file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"columns": [], "nrows": len(df)}

    for col in df.columns:
        name = str(col)
        safe_name = sanitize_filename(name)
        dtype = str(df[col].dtype)
        manifest["columns"].append({"name": name, "file": safe_name, "dtype": dtype})

        if format == "csv":
            path = out / f"{safe_name}.csv"
            df[[col]].to_csv(path, index=True, header=True)
        elif format == "parquet":
            path = out / f"{safe_name}.parquet"
            df[[col]].to_parquet(path, index=True)
        else:
            raise ValueError(f"Unknown format: {format}")

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


if __name__ == "__main__":
    sas_path = "ed2015-sas.sas7bdat"
    df = pd.read_sas(sas_path)

    manifest_path = store_columns(df, out_dir="output_columns", format="csv")
    print(f"Stored {len(df.columns)} columns under output_columns/")
    print(f"Manifest: {manifest_path}")
