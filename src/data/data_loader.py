import zipfile
from pathlib import Path
import pandas as pd


class DataLoader:
    def load_data(self, zip_filename: str):
        project_root = Path(".").resolve()
        raw_dir = project_root / "data" / "raw"
        interim_dir = project_root / "data" / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)

        zip_path = raw_dir / zip_filename

        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        # Extract into interim_dir so raw inputs remain immutable.
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=interim_dir)

        # Find extracted SAS files inside interim_dir
        sas_files = list(interim_dir.glob("*.sas7bdat"))

        if not sas_files:
            raise FileNotFoundError("No .sas7bdat found after extraction.")

        # Pick the largest file (usually correct dataset)
        sas_path = max(sas_files, key=lambda p: p.stat().st_size)

        print(f"Reading SAS file: {sas_path}")

        df = pd.read_sas(sas_path)

        print(f"Loaded shape: {df.shape}")
        return df