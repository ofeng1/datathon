import pandas as pd
from med_proj.common.time import parse_dt, hours_between, days_between
from med_proj.common.logging import get_logger

log = get_logger("labels")

def build_ed_revisit_labels(enc: pd.DataFrame, hours_72: int = 72, days_7: int = 7, days_30: int = 30) -> pd.DataFrame:
    df = enc.copy()

    df["start_dt"] = df["start_time"].apply(parse_dt)
    df["end_dt"] = df["end_time"].apply(parse_dt)
    df = df.sort_values(["patient_id", "start_dt"]).reset_index(drop=True)

    df["label_ed_revisit_72h"] = 0
    df["label_ed_revisit_7d"] = 0
    df["label_ed_revisit_30d"] = 0

    for pid, g in df.groupby("patient_id", sort=False):
        idxs = g.index.to_list()
        for i, cur_i in enumerate(idxs):
            cur_end = df.loc[cur_i, "end_dt"] or df.loc[cur_i, "start_dt"]
            if cur_end is None:
                continue

            for j in range(i + 1, len(idxs)):
                nxt_i = idxs[j]
                nxt_start = df.loc[nxt_i, "start_dt"]
                if nxt_start is None or nxt_start <= cur_end:
                    continue

                gap_h = hours_between(cur_end, nxt_start)
                gap_d = days_between(cur_end, nxt_start)

                if gap_h <= hours_72:
                    df.loc[cur_i, "label_ed_revisit_72h"] = 1
                if gap_d <= days_7:
                    df.loc[cur_i, "label_ed_revisit_7d"] = 1
                if gap_d <= days_30:
                    df.loc[cur_i, "label_ed_revisit_30d"] = 1

                if gap_h > hours_72 and gap_d > days_30:
                    break

    log.info(
        "ED revisit positives: 72h=%d 7d=%d 30d=%d",
        int(df["label_ed_revisit_72h"].sum()),
        int(df["label_ed_revisit_7d"].sum()),
        int(df["label_ed_revisit_30d"].sum()),
    )

    return df.drop(columns=["start_dt", "end_dt"])


def build_ed_to_admit_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'admitted' exists (boolean), create label_ed_admit = 1 when admitted True.
    """
    out = df.copy()
    if "admitted" in out.columns:
        out["label_ed_admit"] = out["admitted"].astype(int)
        log.info("ED->admit positives=%d", int(out["label_ed_admit"].sum()))
    return out