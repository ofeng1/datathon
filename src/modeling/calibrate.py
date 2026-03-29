from sklearn.calibration import CalibratedClassifierCV

def calibrate(model, X, y, method="sigmoid"):
    cal = CalibratedClassifierCV(model, method=method, cv=3)
    cal.fit(X, y)
    return cal