"""Rasa-Sense: AI e-tongue for Dravya identification (hackathon demo).

Run:  source /Users/asitaponbhavya/venvs/python312/bin/activate && python app.py
      then open http://localhost:5050

No hardware needed: readings come from the simulated sensor array in sim.py.
An ESP32 can later POST real readings to /api/ingest (see firmware/).
"""
import csv
import io
import json
import random
import sqlite3
import time

import numpy as np
from flask import Flask, Response, g, jsonify, render_template, request

import ml
import sim

DB_PATH = "etongue.db"
# Left out of the starter dataset on purpose: calibrate it live during the demo.
DEMO_NEW_HERB = "Saindhava"

app = Flask(__name__)
app.json.sort_keys = False
hardware = {"reading": None, "at": 0.0}   # latest reading pushed by an ESP32, if any
mystery = {}                               # hidden sample for "mystery" mode


# ---------------- database ----------------

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    if "db" in g:
        g.db.close()


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript("""
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY, ts TEXT, label TEXT, source TEXT,
        ph REAL, tds REAL, optical REAL, temp REAL, e1 REAL, e2 REAL, e3 REAL, e4 REAL);
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY, ts TEXT, source TEXT, sample_desc TEXT, dravya TEXT,
        confidence REAL, quality REAL, status TEXT, result_json TEXT, reading_json TEXT);
    """)
    if con.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0:
        seed = [(h, r) for h, r in sim.calibration_set() if h != DEMO_NEW_HERB]
        _insert_samples(con, seed, "reference-lab")
    con.commit()
    if ml.load() is None:
        ml.train(_calibration_rows(con))
    con.close()


def _insert_samples(con, rows, source):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    con.executemany(
        "INSERT INTO samples (ts,label,source,ph,tds,optical,temp,e1,e2,e3,e4) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(ts, lbl, source, *[r[k] for k in sim.FEATURES]) for lbl, r in rows])


def _calibration_rows(con):
    rows = con.execute("SELECT * FROM samples").fetchall()
    return [(r["label"], {k: r[k] for k in sim.FEATURES}) for r in rows]


# ---------------- pages ----------------

@app.route("/")
def index():
    return render_template("index.html", page="analyze")


@app.route("/collect")
def collect_page():
    return render_template("collect.html", page="collect")


@app.route("/train")
def train_page():
    return render_template("train.html", page="train")


@app.route("/threshold")
def threshold_page():
    return render_template("threshold.html", page="threshold")


@app.route("/database")
def database_page():
    return render_template("database.html", page="database")


# ---------------- API ----------------

def _hardware_live():
    return hardware["reading"] is not None and time.time() - hardware["at"] < 5


@app.get("/api/meta")
def meta():
    model = ml.load()
    counts = {r["label"]: r["n"] for r in db().execute(
        "SELECT label, COUNT(*) n FROM samples GROUP BY label ORDER BY label")}
    return jsonify({
        "herbs": [{"name": h, "latin": v["latin"], "rasa": v["rasa"]} for h, v in sim.HERBS.items()],
        "adulterants": sim.ADULTERANTS,
        "rasas": sim.RASAS, "rasa_en": sim.RASA_EN,
        "source": "ESP32" if _hardware_live() else "Simulator",
        "calibration_counts": counts,
        "model": None if model is None else {
            "classes": model["classes"], "trained_at": model["trained_at"],
            "n_samples": model["n_samples"], "accuracy": model["accuracy"]},
    })


def _sample_params(src):
    herb = src.get("herb") or None
    if herb == "mystery":
        return "mystery", None, 1.0
    if herb not in sim.HERBS:
        herb = None
    adulterant = src.get("adulterant", "none")
    if adulterant not in sim.ADULTERANTS:
        adulterant = "none"
    conc = min(max(float(src.get("conc", 1.0)), 0.05), 2.0)
    return herb, adulterant, conc


def _read_once(herb, adulterant, conc):
    if _hardware_live():
        return hardware["reading"], "ESP32"
    if herb == "mystery":
        if not mystery:
            return sim.read(None), "Simulator"
        return sim.read(mystery["herb"], mystery["conc"], mystery["adulterant"]), "Simulator"
    return sim.read(herb, conc, adulterant), "Simulator"


@app.get("/api/reading")
def reading():
    r, source = _read_once(*_sample_params(request.args))
    return jsonify({"reading": r, "source": source})


@app.post("/api/mystery")
def new_mystery():
    herbs = ml.load()["classes"]
    mystery.clear()
    mystery.update({
        "herb": random.choice(herbs),
        "adulterant": random.choice(["none", "none", "none"] + [a for a in sim.ADULTERANTS if a != "none"]),
        "conc": round(random.uniform(0.85, 1.15), 2),
    })
    return jsonify({"ok": True})


@app.post("/api/analyze")
def analyze():
    model = ml.load()
    if model is None:
        return jsonify({"error": "No model trained yet"}), 400
    params = request.get_json(force=True)
    herb, adulterant, conc = _sample_params(params)
    reads = [_read_once(herb, adulterant, conc) for _ in range(10)]
    source = reads[0][1]
    avg = {k: round(float(np.mean([r[k] for r, _ in reads])), 3) for k in sim.FEATURES}
    result = ml.analyze(model, avg)

    if source == "ESP32":
        desc = "ESP32 live sample"
    elif herb == "mystery":
        desc = f"Mystery: {mystery['herb']} + {sim.ADULTERANTS[mystery['adulterant']]}"
        result["reveal"] = {"herb": mystery["herb"], "adulterant": sim.ADULTERANTS[mystery["adulterant"]]}
    elif herb is None:
        desc = "Distilled water (blank)"
    else:
        desc = f"{herb} @ {conc:.2f}x, {sim.ADULTERANTS[adulterant]}"
    result["reading"] = avg

    db().execute(
        "INSERT INTO analyses (ts,source,sample_desc,dravya,confidence,quality,status,result_json,reading_json) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), source, desc, result["dravya"], result["confidence"],
         result["quality"], result["status"], json.dumps(result), json.dumps(avg)))
    db().commit()
    return jsonify(result)


@app.post("/api/collect")
def collect():
    """Record calibration readings for an authenticated reference herb."""
    p = request.get_json(force=True)
    label = p.get("label")
    n = min(max(int(p.get("n", 10)), 1), 100)
    if label not in sim.HERBS:
        return jsonify({"error": "Unknown herb"}), 400
    rng = np.random.default_rng()
    rows = [(label, sim.read(label, rng.uniform(0.85, 1.15), "none", rng)) for _ in range(n)]
    _insert_samples(db(), rows, "calibration")
    db().commit()
    return jsonify({"added": n, "label": label, "preview": rows[-1][1]})


@app.post("/make_model")
def make_model():
    rows = _calibration_rows(db())
    try:
        _, report = ml.train(rows)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(report)


@app.get("/api/threshold")
def threshold():
    herb = request.args.get("herb", "Amla")
    if herb not in sim.HERBS:
        return jsonify({"error": "Unknown herb"}), 400
    return jsonify(ml.threshold_study(herb, float(request.args.get("stock", 10))))


@app.get("/api/analyses")
def analyses():
    rows = db().execute("SELECT id,ts,source,sample_desc,dravya,confidence,quality,status "
                        "FROM analyses ORDER BY id DESC LIMIT 200").fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/samples")
def samples():
    rows = db().execute("SELECT * FROM samples ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/export/<table>.csv")
def export(table):
    if table not in ("samples", "analyses"):
        return "Not found", 404
    rows = db().execute(f"SELECT * FROM {table}").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    if rows:
        w.writerow(rows[0].keys())
        w.writerows([tuple(r) for r in rows])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={table}.csv"})


@app.post("/api/reset")
def reset():
    """Restore the starter dataset and model (handy between demo runs)."""
    con = db()
    con.execute("DELETE FROM samples")
    con.execute("DELETE FROM analyses")
    _insert_samples(con, [(h, r) for h, r in sim.calibration_set() if h != DEMO_NEW_HERB], "reference-lab")
    con.commit()
    _, report = ml.train(_calibration_rows(con))
    return jsonify(report)


@app.post("/api/ingest")
def ingest():
    """Future hardware path: ESP32 POSTs {"ph":..,"tds":..,"optical":..,"temp":..,"e1"..."e4"}."""
    r = request.get_json(force=True)
    try:
        hardware["reading"] = {k: float(r[k]) for k in sim.FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": f"Expected fields: {sim.FEATURES}"}), 400
    hardware["at"] = time.time()
    return jsonify({"ok": True})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5050, debug=False)
