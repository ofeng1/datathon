import zipfile
from pathlib import Path

import pandas as pd

class DataLoader:

    def load_data(self, zip_path: str):

        data_dir = Path(".").resolve()
        zip_path = data_dir / "med_proj" / "data"/ zip_path

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=data_dir)

        sas_path = data_dir / "ed2015-sas.sas7bdat"
        df = pd.read_sas(sas_path)

        return df

if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data("ed2015-sas.sas7bdat.zip")
