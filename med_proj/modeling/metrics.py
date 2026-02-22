from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def metrics_binary(y_true, y_prob):
    out = {}
    out["auroc"] = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    out["auprc"] = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    out["brier"] = brier_score_loss(y_true, y_prob)
    return out