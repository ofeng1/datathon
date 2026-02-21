from pathlib import Path
import zipfile, tempfile
import pandas as pd

def load_sas_data(zip_path):
    sas_member_suffix = ".sas7bdat"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(zip_path) as z:
            sas_name = next(n for n in z.namelist() if n.lower().endswith(sas_member_suffix))
            sas_path = z.extract(sas_name, path=td)

        df = pd.read_sas(sas_path, format="sas7bdat", encoding="latin-1") 

    return df