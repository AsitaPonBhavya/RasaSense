"""ML pipeline: Dravya classifier, Rasa regressor, adulteration detector."""
import pickle
import time

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sim

MODEL_PATH = "model.pkl"
# Adulterated samples deviate up to ~12x the acceptance limit; a herb absent from
# the library deviates ~20x+. Beyond this factor we call it unknown, not adulterated.
UNKNOWN_FACTOR = 15

# numpy 2 + macOS Accelerate emits spurious matmul warnings; results are fine.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def features(r):
    """Sensor dict -> feature vector, with TDS temperature-compensated to 25 degC."""
    tds25 = r["tds"] / (1 + 0.02 * (r["temp"] - 25.0))
    return [r["ph"], tds25, r["optical"], r["e1"], r["e2"], r["e3"], r["e4"]]


def train(rows):
    """rows: list of (label, reading_dict). Returns (model, report)."""
    labels = np.array([str(lbl) for lbl, _ in rows])
    X = np.array([features(r) for _, r in rows])
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("Need calibration samples from at least 2 herbs")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Hold-out accuracy for the report, then refit on everything.
    Xtr, Xte, ytr, yte = train_test_split(Xs, labels, test_size=0.25, random_state=0, stratify=labels)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    accuracy = float((clf.predict(Xte) == yte).mean())
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xs, labels)

    # Rasa regressor: sensors -> six taste intensities (targets from reference rasa profiles).
    known = [i for i, lbl in enumerate(labels) if lbl in sim.HERBS]
    rasa_reg = None
    if known:
        Y = np.array([sim.HERBS[labels[i]]["rasa"] for i in known])
        rasa_reg = Ridge(alpha=1.0).fit(Xs[known], Y)

    # Adulteration detector: Mahalanobis distance to the predicted herb's centroid,
    # using the pooled within-class covariance of authentic samples.
    centroids = {c: Xs[labels == c].mean(axis=0) for c in classes}
    resid = np.vstack([Xs[labels == c] - centroids[c] for c in classes])
    cov = np.cov(resid, rowvar=False) + 1e-3 * np.eye(Xs.shape[1])
    inv_cov = np.linalg.inv(cov)
    limits = {}
    for c in classes:
        d = _mahal(Xs[labels == c], centroids[c], inv_cov)
        limits[c] = {"p50": float(np.percentile(d, 50)), "limit": float(np.percentile(d, 99)) * 1.15}

    model = {
        "scaler": scaler, "clf": clf, "rasa": rasa_reg, "centroids": centroids,
        "inv_cov": inv_cov, "limits": limits, "classes": classes,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_samples": len(rows),
        "accuracy": accuracy,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model, {"accuracy": accuracy, "n_samples": len(rows), "classes": [str(c) for c in classes],
                   "trained_at": model["trained_at"]}


def load():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


def _mahal(Xs, centroid, inv_cov):
    diff = np.atleast_2d(Xs) - centroid
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))


def analyze(model, reading):
    xs = model["scaler"].transform([features(reading)])
    proba = model["clf"].predict_proba(xs)[0]
    order = np.argsort(proba)[::-1]
    cls = model["clf"].classes_
    pred, conf = str(cls[order[0]]), float(proba[order[0]])

    rasa = [0.0] * 6
    if model["rasa"] is not None:
        rasa = np.clip(model["rasa"].predict(xs)[0], 0, 1).round(3).tolist()

    d = float(_mahal(xs, model["centroids"][pred], model["inv_cov"])[0])
    lim = model["limits"][pred]
    # Quality: 100 near the reference centre, 60 at the acceptance limit, 0 at 2x limit.
    if d <= lim["p50"]:
        quality = 100.0
    elif d <= lim["limit"]:
        quality = 100 - 40 * (d - lim["p50"]) / (lim["limit"] - lim["p50"])
    else:
        quality = max(0.0, 60 - 60 * (d - lim["limit"]) / lim["limit"])

    alerts = []
    if reading["tds"] < 30 and reading["optical"] < 0.06:
        status = "NO SAMPLE"
        alerts.append("Sensor response matches distilled water. Dip the probe in a herbal extract.")
    elif d > UNKNOWN_FACTOR * lim["limit"]:
        status = "UNKNOWN DRAVYA"
        alerts.append("Fingerprint matches no herb in the reference library. Calibrate it as a new Dravya, "
                      "or treat as a substitute / spurious sample.")
    elif d > lim["limit"]:
        status = "ADULTERATION SUSPECTED"
        alerts.append(f"Fingerprint deviates {d / lim['limit']:.1f}x beyond the accepted limit for {pred}.")
        alerts += _explain_deviation(model, xs[0], pred)
    elif conf < 0.6:
        status = "LOW CONFIDENCE"
    else:
        status = "WITHIN THRESHOLD"
    if conf < 0.6 and status != "NO SAMPLE":
        alerts.append(f"Classifier is unsure (top match {conf:.0%}); sample may be a mixture or an untrained herb.")

    return {
        "dravya": pred,
        "latin": sim.HERBS.get(pred, {}).get("latin", ""),
        "confidence": round(conf, 3),
        "top3": [{"herb": str(cls[i]), "p": round(float(proba[i]), 3)} for i in order[:3]],
        "rasa": rasa,
        "reference_rasa": sim.HERBS.get(pred, {}).get("rasa"),
        "distance": round(d, 2),
        "limit": round(lim["limit"], 2),
        "quality": round(quality, 1),
        "status": status,
        "alerts": alerts,
    }


_FEATURE_HINTS = {
    0: ("pH", "pH higher than reference: possible alkaline filler (chalk / lime)", "pH lower than reference"),
    1: ("TDS", "Dissolved solids higher than reference: added minerals or salts", "Dissolved solids lower: diluted / weak sample"),
    2: ("Optical", "Absorbance higher than reference: added colour or turbid filler (dye / starch)", "Absorbance lower: weak or diluted extract"),
    3: ("Bitter electrode", "Bitter response higher than reference", "Bitter response weak: low active-principle content"),
    4: ("Sweet electrode", "Sweet response higher than reference: possible added sugar / starch", "Sweet response lower than reference"),
    5: ("Pungent electrode", "Pungent response higher than reference", "Pungent response weak: low volatile-oil content"),
    6: ("Astringent electrode", "Astringent response higher than reference", "Astringent response weak: low tannin content"),
}


def _explain_deviation(model, x, pred):
    z = x - model["centroids"][pred]
    hints = []
    for i in np.argsort(-np.abs(z))[:2]:
        if abs(z[i]) < 1.5:
            continue
        name, hi, lo = _FEATURE_HINTS[i]
        hints.append(hi if z[i] > 0 else lo)
    return hints


# --- Threshold detection: dilution series -> minimum detectable concentration ---

RASA_CHANNEL = {  # rasa index -> (feature name, sign)
    0: ("e2", 1), 1: ("ph", -1), 2: ("tds", 1), 3: ("e3", 1), 4: ("e1", 1), 5: ("e4", 1),
}


def threshold_study(herb, stock_mg_ml=10.0, steps=12, reps=3):
    rng = np.random.default_rng()
    blanks = [sim.read(None, rng=rng) for _ in range(20)]
    concs = [stock_mg_ml / 2 ** i for i in range(steps)]
    series = []
    for c in concs:
        reads = [sim.read(herb, c / stock_mg_ml, rng=rng) for _ in range(reps)]
        series.append({k: float(np.mean([r[k] for r in reads])) for k in sim.FEATURES})

    rasa_ref = sim.HERBS[herb]["rasa"]
    out = []
    for i, (feat, sign) in RASA_CHANNEL.items():
        b = np.array([r[feat] for r in blanks])
        lod = 3 * b.std()
        signal = [sign * (s[feat] - b.mean()) for s in series]
        thr = None
        for c, s in zip(concs, signal):   # walk from strongest to weakest
            if s > lod:
                thr = c
            else:
                break
        out.append({
            "rasa": sim.RASAS[i], "en": sim.RASA_EN[i], "channel": feat,
            "lod_signal": round(float(lod), 3),
            "signal": [round(float(s), 3) for s in signal],
            "threshold_mg_ml": None if thr is None or rasa_ref[i] < 0.1 else round(thr, 4),
        })
    return {"herb": herb, "concentrations": [round(c, 4) for c in concs], "channels": out}
