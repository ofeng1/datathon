import zipfile
from pathlib import Path
import pandas as pd


class DataLoader:

    def load_data(self, zip_path: str):

        data_dir = Path(".").resolve()
        zip_path = data_dir / zip_path

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=data_dir)

        # Find extracted SAS files inside data_dir
        sas_files = list(data_dir.glob("*.sas7bdat"))

        if not sas_files:
            raise FileNotFoundError("No .sas7bdat found after extraction.")

        # Pick the largest file (usually correct dataset)
        sas_path = max(sas_files, key=lambda p: p.stat().st_size)

        print(f"Reading SAS file: {sas_path}")

        df = pd.read_sas(sas_path)

        return df