import numpy as np
import pandas as pd

from med_proj.common.time import parse_dt, hours_between, days_between
from med_proj.common.logging import get_logger

log = get_logger("features")

def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["start_dt"] = df["start_time"].apply(parse_dt)
    df["end_dt"] = df["end_time"].apply(parse_dt)
    df = df.sort_values(["patient_id", "start_dt"]).reset_index(drop=True)

    def _los(row):
        a = row["start_dt"]
        b = row["end_dt"]
        if a is None or b is None:
            return 0.0
        return max(0.0, hours_between(a, b))

    df["los_hours"] = df.apply(_los, axis=1)
    df["encounter_hour"] = df["start_dt"].apply(lambda x: x.hour if x else -1)
    df["encounter_dow"] = df["start_dt"].apply(lambda x: x.weekday() if x else -1)

    df["prior_ed_30d"] = 0
    df["prior_ed_180d"] = 0
    df["days_since_last_encounter"] = np.nan

    for pid, g in df.groupby("patient_id", sort=False):
        idxs = g.index.to_list()
        times = df.loc[idxs, "start_dt"].tolist()

        for k, cur_i in enumerate(idxs):
            cur_t = times[k]
            if k == 0 or cur_t is None or times[k - 1] is None:
                df.loc[cur_i, "days_since_last_encounter"] = 999.0
            else:
                df.loc[cur_i, "days_since_last_encounter"] = max(0.0, days_between(times[k - 1], cur_t))

            ed30 = 0
            ed180 = 0
            for m in range(0, k):
                prev_t = times[m]
                if prev_t is None or cur_t is None:
                    continue
                dd = days_between(prev_t, cur_t)
                if dd <= 30:
                    ed30 += 1
                if dd <= 180:
                    ed180 += 1
            df.loc[cur_i, "prior_ed_30d"] = ed30
            df.loc[cur_i, "prior_ed_180d"] = ed180

    df = df.drop(columns=["start_dt", "end_dt"])
    log.info("Built features rows=%d", len(df))
    return df