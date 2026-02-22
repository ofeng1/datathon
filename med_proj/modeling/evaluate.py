from med_proj.modeling.metrics import metrics_binary
from med_proj.common.logging import get_logger

log = get_logger("evaluate")

def evaluate(model, X_test, y_test, name: str):
    prob = model.predict_proba(X_test)[:, 1]
    m = metrics_binary(y_test, prob)
    log.info("[%s] AUROC=%.4f AUPRC=%.4f Brier=%.4f", name, m["auroc"], m["auprc"], m["brier"])
    return m