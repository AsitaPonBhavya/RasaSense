"""Simulated e-tongue sensor array.

Stands in for the ESP32 hardware during the demo. Each herb has an Ayurvedic
Rasa profile; sensor responses are generated from that profile so the
readings are physically plausible (e.g. sour herbs drop pH, salty raise TDS).

Sensor channels (same as the ESP32 firmware sends):
  ph       - pH probe
  tds      - conductivity / TDS probe (ppm)
  optical  - absorbance from LED + photodiode (0..1)
  temp     - DS18B20 temperature (deg C)
  e1..e4   - electrochemical electrode array (mV), each with a lipid/polymer
             membrane tuned towards a taste class:
             e1 bitter, e2 sweet, e3 pungent, e4 astringent
"""
import numpy as np

RASAS = ["Madhura", "Amla", "Lavana", "Katu", "Tikta", "Kashaya"]
RASA_EN = ["Sweet", "Sour", "Salty", "Pungent", "Bitter", "Astringent"]
FEATURES = ["ph", "tds", "optical", "temp", "e1", "e2", "e3", "e4"]

# Rasa order: sweet, sour, salty, pungent, bitter, astringent
HERBS = {
    "Tulsi":       {"latin": "Ocimum tenuiflorum",     "rasa": [0.05, 0.0, 0.0, 0.80, 0.60, 0.15]},
    "Neem":        {"latin": "Azadirachta indica",     "rasa": [0.0, 0.0, 0.0, 0.10, 0.95, 0.60]},
    "Amla":        {"latin": "Phyllanthus emblica",    "rasa": [0.30, 0.95, 0.0, 0.20, 0.30, 0.70]},
    "Ashwagandha": {"latin": "Withania somnifera",     "rasa": [0.35, 0.0, 0.0, 0.10, 0.70, 0.55]},
    "Haridra":     {"latin": "Curcuma longa",          "rasa": [0.0, 0.0, 0.0, 0.70, 0.75, 0.20]},
    "Shunthi":     {"latin": "Zingiber officinale",    "rasa": [0.20, 0.0, 0.0, 0.95, 0.10, 0.05]},
    "Yashtimadhu": {"latin": "Glycyrrhiza glabra",     "rasa": [0.95, 0.0, 0.0, 0.0, 0.15, 0.10]},
    "Saindhava":   {"latin": "Rock salt (mineral)",    "rasa": [0.15, 0.0, 0.95, 0.0, 0.0, 0.0]},
}

ADULTERANTS = {
    "none":   "None (authentic)",
    "starch": "Starch filler",
    "chalk":  "Chalk / CaCO3",
    "dye":    "Synthetic colour",
    "sugar":  "Added sugar",
}

# Each herb also has a small fixed "fingerprint" beyond its rasa (colour,
# minerals, specific phytochemicals) so herbs with similar rasa stay separable.
_rng = np.random.default_rng(42)
_FINGERPRINT = {h: _rng.normal(0, 1, len(FEATURES)) for h in HERBS}
_HERB_OPTICAL = {"Haridra": 0.30, "Neem": 0.12, "Tulsi": 0.10, "Amla": 0.05,
                 "Ashwagandha": 0.08, "Shunthi": 0.06, "Yashtimadhu": 0.10, "Saindhava": -0.10}

BLANK_NOISE = np.array([0.04, 6.0, 0.006, 0.15, 1.5, 1.5, 1.5, 1.5])


def ideal_response(herb, conc=1.0, adulterant="none", temp=25.0):
    """Noise-free sensor vector for a herb extract at relative concentration conc."""
    sw, so, sa, pu, bi, ast = HERBS[herb]["rasa"]
    c = conc
    if adulterant == "starch":       # filler dilutes the active herb
        c = conc * 0.55
    fp = _FINGERPRINT[herb]

    ph = 7.0 - c * (3.6 * so + 0.5 * ast + 0.2 * bi) + c * 0.15 * fp[0]
    tds = 4 + c * (56 + 1100 * sa + 260 * ast + 220 * so + 140 * sw + 110 * pu + 120 * bi) + c * 25 * fp[1]
    optical = 0.02 + c * (0.19 + 0.35 * ast + 0.22 * bi + _HERB_OPTICAL[herb]) + c * 0.02 * fp[2]
    e1 = 10 + c * (8 + 190 * bi + 35 * ast) + c * 8 * fp[4]
    e2 = 10 + c * (4 + 175 * sw + 10 * so) + c * 8 * fp[5]
    e3 = 10 + c * (2 + 185 * pu + 25 * bi) + c * 8 * fp[6]
    e4 = 10 + c * (5 + 165 * ast + 45 * so) + c * 8 * fp[7]

    if adulterant == "starch":
        optical += 0.22 * conc            # turbidity
        e2 += 25 * conc
    elif adulterant == "chalk":
        ph += 1.6 * conc
        tds += 320 * conc
        e4 -= 30 * conc
    elif adulterant == "dye":
        optical += 0.38 * conc
        e1 += 20 * conc
    elif adulterant == "sugar":
        e2 += 110 * conc
        tds += 40 * conc

    # Conductivity rises ~2%/degC; the app compensates back to 25 degC.
    tds *= 1 + 0.02 * (temp - 25.0)
    return np.array([ph, tds, min(optical, 1.5), temp, e1, e2, e3, e4])


def read(herb=None, conc=1.0, adulterant="none", rng=None):
    """One noisy sensor reading. herb=None simulates distilled water (blank)."""
    rng = rng or np.random.default_rng()
    temp = 25.0 + rng.normal(0, 0.8)
    if herb is None:
        base = np.array([7.0, 4.0, 0.02, temp, 10, 10, 10, 10], dtype=float)
    else:
        base = ideal_response(herb, conc, adulterant, temp)
    noise = BLANK_NOISE.copy()
    noise[1] += 0.015 * base[1]        # TDS noise scales with signal
    out = base + rng.normal(0, noise)
    out[3] = temp
    return {k: round(float(v), 3) for k, v in zip(FEATURES, out)}


def calibration_set(per_herb=40, seed=7):
    """Synthetic 'authenticated reference sample' dataset used to seed the DB."""
    rng = np.random.default_rng(seed)
    rows = []
    for herb in HERBS:
        for _ in range(per_herb):
            # authentic batches vary a little in strength (batch-to-batch variation)
            conc = rng.uniform(0.85, 1.15)
            rows.append((herb, read(herb, conc, "none", rng)))
    return rows
