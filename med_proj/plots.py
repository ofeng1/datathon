import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import DataLoader
import pandas as pd

data_loader = DataLoader()
zip_paths = ["ed2015-sas.sas7bdat.zip", "ed2016_sas.zip", "ed2017_sas.zip", "ed2018_sas.zip",
                "ed2019_sas.zip", "ed2020_sas.zip", "ed2021_sas.zip"]
df = pd.DataFrame()
for zip_path in zip_paths:
    df_temp = data_loader.load_data(zip_path)
    df = pd.concat([df, df_temp])

plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, color='blue', edgecolor='black')
plt.show()