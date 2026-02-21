import zipfile
from pathlib import Path

import pandas as pd

class DataLoader:

    def load_data(self, zip_path: str):

        data_dir = Path(".").resolve()
        zip_path = data_dir / zip_path

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=data_dir)

        sas_path = data_dir / "ed2015-sas.sas7bdat"
        df = pd.read_sas(sas_path)

        return df

if __name__ == "__main__":
    data_loader = DataLoader()
    zip_paths = ["ed2015-sas.sas7bdat.zip", "ed2016_sas.zip", "ed2017_sas.zip", "ed2018_sas.zip",
                 "ed2019_sas.zip", "ed2020_sas.zip", "ed2021_sas.zip"]
    df = pd.DataFrame()
    for zip_path in zip_paths:
        df_temp = data_loader.load_data(zip_path)
        df = pd.concat([df, df_temp])
    print(df.shape)