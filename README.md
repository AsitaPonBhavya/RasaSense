# Rasa-Sense: AI E-Tongue for Dravya Identification

**Problem statement:** SAFH25025, E-tongue for Dravya identification (Hardware · MedTech / BioTech / HealthTech)

Rasa-Sense is a sensor-integrated, AI-enabled electronic tongue. It is designed to:

1. **Identify** an Ayurvedic herb (Dravya) from its chemical "taste fingerprint"
2. **Estimate its six Rasas** (Shad-Rasa: sweet, sour, salty, pungent, bitter, astringent)
3. **Detect adulteration or substandard quality**, with a plain-language reason
4. **Recognise unknown herbs** instead of forcing a wrong label
5. **Measure taste thresholds**: the minimum detectable concentration of each Rasa
6. **Keep learning**: every calibration sample goes into a database, and the model can be retrained at any time

> ⚠️ **Demo mode (no hardware):** this build runs on a **simulated sensor array** (`sim.py`). The ML models, alerts,
> database and threshold logic are real code. Only the sensor readings are generated. The dashboard shows a
> "Simulated sensor array" badge. When an ESP32 sends real readings to `/api/ingest`, the badge switches to
> "ESP32 live" and the same pipeline runs on real data.

---

## Table of contents
1. [How to run](#1-how-to-run)
2. [System overview](#2-system-overview)
3. [The sensor array](#3-the-sensor-array-8-channels)
4. [The simulator](#4-the-simulator-simpy)
5. [Data pipeline and preprocessing](#5-data-pipeline-and-preprocessing)
6. [AI / ML models used](#6-ai--ml-models-used)
7. [Decision logic: final status and alerts](#7-decision-logic-final-status-and-alerts)
8. [Threshold detection module](#8-threshold-detection-module)
9. [Database](#9-database)
10. [Web dashboard pages](#10-web-dashboard-pages)
11. [API reference](#11-api-reference)
12. [Hardware design (ESP32)](#12-hardware-design-esp32)
13. [Demo script for judges](#13-demo-script-for-judges)
14. [Questions judges may ask](#14-questions-judges-may-ask)
15. [Limitations (be honest about these)](#15-limitations-be-honest-about-these)
16. [Roadmap](#16-roadmap)
17. [Tech stack and file structure](#17-tech-stack-and-file-structure)

---

## 1. How to run

```bash
source /Users/asitaponbhavya/venvs/python312/bin/activate
cd ~/Desktop/Try
pip install -r requirements.txt      # first time only (already installed)
python app.py
```

Open **http://localhost:5050** in a browser.

- On the first run the app creates `etongue.db` (SQLite), seeds it with 280 reference samples (7 herbs × 40) and trains the model (`model.pkl`).
- Everything works **offline**. Chart.js is bundled in `static/`; only the Poppins font comes from the internet, and a fallback font is used without it.
- Port 5050 is used because macOS's AirPlay Receiver often occupies port 5000.
- To start completely fresh, stop the app, delete `etongue.db` and `model.pkl`, and run it again.

---

## 2. System overview

```
 ┌──────────────┐   ┌───────────────────────┐   ┌──────────────────┐   ┌────────────────────────────┐   ┌───────────────┐
 │ Herbal       │   │ Multi-sensor e-tongue  │   │ ESP32            │   │ AI / ML (Python, sklearn)  │   │ Dashboard     │
 │ extract in   │──▶│ pH · TDS · Optical ·   │──▶│ ADC, averaging,  │──▶│ 1. Random Forest  → herb   │──▶│ result, Rasa  │
 │ a beaker     │   │ Temp · 4 electrodes    │   │ Wi-Fi JSON POST  │   │ 2. Ridge          → 6 Rasa │   │ radar, alerts │
 └──────────────┘   └───────────────────────┘   └──────────────────┘   │ 3. Mahalanobis    → quality│   │ database, CSV │
                       (simulated in demo)        (sim.py in demo)     │ 4. Threshold study (LOD)   │   └───────────────┘
                                                                       └────────────────────────────┘
                                                                                     ▲
                                                                     SQLite database ┘ (calibration + analysis log)
```

**One analysis, step by step:**
1. Take **10 sensor readings** of the sample and average them to reduce noise.
2. **Temperature-compensate** the TDS reading.
3. **Standardise** the features using the training statistics.
4. **Random Forest** predicts which herb it is, with a probability for each herb.
5. **Ridge regression** estimates the six Rasa intensities.
6. **Mahalanobis distance** measures how far the sample is from authentic samples of that herb, which gives the quality score and the adulteration / unknown decision.
7. The **explanation engine** names the sensors that deviated most and translates them into likely causes.
8. The result is **saved to the database** and shown on the dashboard.

---

## 3. The sensor array (8 channels)

| # | Channel | Physical sensor | Unit | What it responds to | Main Rasa link |
|---|---|---|---|---|---|
| 1 | `ph` | pH probe (PH-4502C module) | pH | Acidity (organic acids such as ascorbic, gallic) | **Amla** (sour) lowers pH |
| 2 | `tds` | Conductivity / TDS probe | ppm | Dissolved ions, salts, minerals | **Lavana** (salty) raises TDS |
| 3 | `optical` | 450 nm LED + photodiode through the sample | Absorbance (AU) | Colour, pigments, tannins, turbidity | **Kashaya / Tikta** (astringent / bitter) |
| 4 | `temp` | DS18B20 | °C | Sample temperature, used for compensation, not as a feature | n/a |
| 5 | `e1` | Electrode with a lipid membrane tuned to bitter compounds | mV | Alkaloids, bitter glycosides | **Tikta** (bitter) |
| 6 | `e2` | Electrode tuned to sweet compounds | mV | Sugars, glycyrrhizin | **Madhura** (sweet) |
| 7 | `e3` | Electrode tuned to pungent compounds | mV | Gingerols, volatile oils, piperine | **Katu** (pungent) |
| 8 | `e4` | Electrode tuned to astringent compounds | mV | Tannins, polyphenols | **Kashaya** (astringent) |

The electrodes are **cross-sensitive**: each responds mainly to one taste but a little to others, as real e-tongue
sensors do. That's why we use ML on the whole 8-channel pattern instead of reading each sensor on its own.

---

## 4. The simulator (`sim.py`)

Because there is no hardware, `sim.py` produces **physically plausible** readings.

### 4.1 Herb library (Rasa profiles, 0 to 1)

Based on the classical Ayurvedic Rasa of each Dravya:

| Herb (Dravya) | Botanical name | Sweet | Sour | Salty | Pungent | Bitter | Astringent |
|---|---|---|---|---|---|---|---|
| Tulsi | *Ocimum tenuiflorum* | 0.05 | 0 | 0 | **0.80** | 0.60 | 0.15 |
| Neem | *Azadirachta indica* | 0 | 0 | 0 | 0.10 | **0.95** | 0.60 |
| Amla | *Phyllanthus emblica* | 0.30 | **0.95** | 0 | 0.20 | 0.30 | 0.70 |
| Ashwagandha | *Withania somnifera* | 0.35 | 0 | 0 | 0.10 | **0.70** | 0.55 |
| Haridra (turmeric) | *Curcuma longa* | 0 | 0 | 0 | 0.70 | **0.75** | 0.20 |
| Shunthi (dry ginger) | *Zingiber officinale* | 0.20 | 0 | 0 | **0.95** | 0.10 | 0.05 |
| Yashtimadhu (licorice) | *Glycyrrhiza glabra* | **0.95** | 0 | 0 | 0 | 0.15 | 0.10 |
| Saindhava (rock salt) | mineral | 0.15 | 0 | **0.95** | 0 | 0 | 0 |

Saindhava is **deliberately left out of the starting database**, so you can show the system learning a new Dravya live (demo step 5).

### 4.2 How a reading is generated

For herb Rasa vector *(sweet, sour, salty, pungent, bitter, astringent)* at relative concentration *c*:

- **pH** = 7.0 − c·(3.6·sour + 0.5·astringent + 0.2·bitter)
- **TDS** = 4 + c·(56 + 1100·salty + 260·astringent + 220·sour + 140·sweet + 110·pungent + 120·bitter)
- **Optical** = 0.02 + c·(0.19 + 0.35·astringent + 0.22·bitter + herb colour term). Haridra gets the largest colour term (curcumin is yellow).
- **E1 bitter** = 10 + c·(8 + 190·bitter + 35·astringent)
- **E2 sweet** = 10 + c·(4 + 175·sweet + 10·sour)
- **E3 pungent** = 10 + c·(2 + 185·pungent + 25·bitter)
- **E4 astringent** = 10 + c·(5 + 165·astringent + 45·sour)

Additions:
- A **fixed herb-specific fingerprint offset** (random, seeded), which stands in for each herb's unique minerals and phytochemicals. It keeps herbs with similar Rasa separable.
- **Temperature effect:** conductivity rises 2% per °C away from 25 °C. Sample temperature varies 25 ± 0.8 °C.
- **Gaussian sensor noise** on every channel; TDS noise grows with signal.
- **Blank (distilled water):** pH 7, TDS ≈ 4 ppm, optical ≈ 0.02, electrodes ≈ 10 mV.

### 4.3 Adulterants simulated

| Adulterant | What it represents | Effect on sensors |
|---|---|---|
| Starch filler | Bulking with cheap starch / flour | Active herb diluted to 55%; turbidity raises optical by +0.22; sweet electrode +25 mV |
| Chalk / CaCO₃ | Alkaline mineral filler | pH +1.6, TDS +320 ppm, astringent electrode −30 mV |
| Synthetic colour | Dye added to fake colour (e.g. in turmeric) | Optical +0.38, bitter electrode +20 mV |
| Added sugar | Sugar syrup to mask taste / add weight | Sweet electrode +110 mV, TDS +40 ppm |

### 4.4 Calibration dataset
`calibration_set()` produces **40 samples per herb** from authentic material. Strength varies randomly between
0.85× and 1.15× to mimic **batch-to-batch variation** between suppliers and harvests.

---

## 5. Data pipeline and preprocessing

| Step | What | Why |
|---|---|---|
| Averaging | Mean of 10 consecutive readings per analysis | Reduces random sensor noise |
| Temperature compensation | `TDS₂₅ = TDS / (1 + 0.02·(T − 25))` | Conductivity depends on temperature; samples measured at 22 °C and 28 °C must compare fairly |
| Feature vector | `[pH, TDS₂₅, optical, E1, E2, E3, E4]`: 7 features | Temperature is used only for compensation, not as an identity feature |
| Standardisation | `StandardScaler`: each feature scaled to mean 0, std 1 | Sensors have very different units (pH ≈ 5, TDS ≈ 500); without scaling the large numbers would dominate |

---

## 6. AI / ML models used

All models come from **scikit-learn** and are trained together in `ml.train()`. They are saved as one bundle
in `model.pkl` and replaced on every retrain.

### Model 1: StandardScaler (preprocessing)
- **Type:** feature standardisation, `sklearn.preprocessing.StandardScaler`
- **Fitted on:** all calibration samples
- **Role:** puts all 7 features on the same scale for the other models

### Model 2: Random Forest Classifier (Dravya identification)
- **Type:** ensemble of decision trees, `sklearn.ensemble.RandomForestClassifier`
- **Settings:** `n_estimators=200` trees, `random_state=0`
- **Input:** 7 standardised features
- **Output:** the most likely herb, plus a **probability for every herb** (the "AI confidence" ring and the "Closest matches" bars)
- **Why Random Forest:**
  - Works well on small tabular datasets (a few hundred samples)
  - Handles non-linear sensor interactions and cross-sensitivity
  - Gives class probabilities, used as confidence
  - Robust to noise; needs no feature engineering
  - Fast enough to run in milliseconds, and small enough to port to an edge device later
- **Validation:** a stratified **75 / 25 train-test split** reports the hold-out accuracy shown on the Train page. The model is then refit on 100% of the data for deployment.
- **Current hold-out accuracy:** 100% on the simulated data (8 classes). Expect lower on real sensors.

### Model 3: Ridge Regression (Shad-Rasa estimation)
- **Type:** linear regression with L2 regularisation, `sklearn.linear_model.Ridge(alpha=1.0)`, multi-output
- **Input:** 7 standardised features
- **Output:** 6 numbers, the intensities of Madhura, Amla, Lavana, Katu, Tikta and Kashaya, clipped to 0–1
- **Training targets:** the reference Rasa profile of each sample's herb (table 4.1)
- **Role:** draws the green "Measured" shape on the radar chart, compared with the dashed "Reference" shape for the identified herb
- **Why Ridge:** taste intensity relates roughly linearly to sensor response. Ridge is stable when sensors are correlated and is easy to explain.
- **Use case:** a sample's taste profile can be measured **objectively** instead of by a human taster. This addresses the subjectivity problem in the problem statement.

### Model 4: Mahalanobis-distance anomaly detector (quality and adulteration)
- **Type:** statistical one-class / outlier model. It is custom code in `ml.py` built with NumPy, not a scikit-learn estimator.
- **How it is trained:**
  1. For each herb, compute its **centroid**: the average fingerprint of authentic samples.
  2. Compute the **pooled within-class covariance** (how sensors naturally vary together inside one herb), plus a small regularisation term (1e-3·I) for numerical stability, then invert it.
  3. For each herb, compute the distances of its own training samples to its centroid and store:
     - `p50`: the median distance (a typical authentic sample)
     - `limit`: the **99th percentile × 1.15**, the acceptance limit
- **How it is used:** for a new sample, compute the Mahalanobis distance to the centroid of the **predicted herb**:

  `d = √((x − μ)ᵀ Σ⁻¹ (x − μ))`

  Mahalanobis distance is better than plain distance because it accounts for sensor correlations and natural spread. A small change in a very stable sensor counts more than a large change in a noisy one.
- **Why it's needed:** a classifier always picks *some* herb, even for fake material. The anomaly detector answers a different question: does this sample look like genuine material of that herb?
- **Quality score (0–100)**, computed from `d`:
  - `d ≤ p50` → **100**
  - `p50 < d ≤ limit` → linear from 100 down to **60**
  - `d > limit` → linear from 60 down to **0** at 2× the limit

### Model 5: Explanation engine (rule-based, likely cause of adulteration)
- **Type:** interpretable rule layer on top of Model 4
- **How:** compute the per-feature deviation (z-score) between the sample and the herb's centroid. Take the **2 largest deviations** with |z| > 1.5 and map each, by direction, to a likely cause:

| Feature deviation | Message shown |
|---|---|
| pH ↑ | Possible alkaline filler (chalk / lime) |
| pH ↓ | pH lower than reference |
| TDS ↑ | Added minerals or salts |
| TDS ↓ | Diluted / weak sample |
| Optical ↑ | Added colour or turbid filler (dye / starch) |
| Optical ↓ | Weak or diluted extract |
| Bitter electrode ↓ | Low active-principle content |
| Sweet electrode ↑ | Possible added sugar / starch |
| Pungent electrode ↓ | Low volatile-oil content |
| Astringent electrode ↓ | Low tannin content |

- **Important:** this is an **informed hypothesis, not proof** of the specific adulterant. The system reliably says the sample is not authentic; naming the adulterant is a likely explanation that a lab test (HPTLC / LC-MS) would confirm. See section 14.

### Model 6: Threshold / limit-of-detection model
A statistical procedure (no learned weights), described in section 8.

### Summary table

| # | Model | Library | Task | Output on screen |
|---|---|---|---|---|
| 1 | StandardScaler | scikit-learn | Preprocessing | n/a |
| 2 | Random Forest (200 trees) | scikit-learn | Classification: which herb | Identified Dravya, confidence ring, closest matches |
| 3 | Ridge regression (α = 1) | scikit-learn | Regression: 6 Rasa intensities | Shad-Rasa radar chart |
| 4 | Mahalanobis anomaly detector | NumPy | One-class outlier detection | Quality score, deviation bar, status badge |
| 5 | Deviation-to-cause rules | Python | Explainability | Alert messages |
| 6 | LOD (3σ) dilution analysis | NumPy | Threshold detection | Threshold page table and curves |

---

## 7. Decision logic: final status and alerts

Rules are checked in this order (`ml.analyze()`):

| Order | Condition | Status shown | Meaning |
|---|---|---|---|
| 1 | TDS < 30 ppm **and** optical < 0.06 | ○ **No sample detected** | Probe is in plain water |
| 2 | d > **15 ×** limit | ✖ **Unknown Dravya: not in library** | Fingerprint matches no known herb. It's a new herb to calibrate, or a substitute / spurious material |
| 3 | d > limit | ⚠ **Adulteration suspected** | Looks like the herb, but outside the authentic range |
| 4 | Random Forest confidence < 60% | ? **Low confidence** | Possibly a mixture or a borderline sample |
| 5 | otherwise | ✔ **Within threshold: Authentic** | Passes |

An extra alert is added whenever confidence is below 60%.

**Why the factor 15 for "unknown":** in testing, adulterated samples deviated **3× to 12×** beyond the limit, while a
herb missing from the library (Saindhava) deviated about **23×**. A cut-off of 15 separates the two cases cleanly.

**Measured behaviour on simulated data** (10 runs × every herb):

| Scenario | Result |
|---|---|
| Authentic samples | 0 / 80 false alarms; 100% correctly identified |
| + Starch / Chalk / Dye / Sugar | 80 / 80 flagged for every adulterant |
| Herb identity under heavy starch or chalk | Sometimes misnamed (~14%), but still flagged as adulterated |
| Untrained herb (Saindhava) | "Unknown Dravya" |
| Distilled water | "No sample detected" |

---

## 8. Threshold detection module

Covers the problem statement's requirement for a *"threshold detection module for quantifying the minimum
perceptible concentration of each taste-related compound"*.

**Procedure (`ml.threshold_study()`):**
1. Measure **20 blank** (distilled water) readings to get the mean and standard deviation (σ) of each channel.
2. **Limit of detection (LOD) = 3σ** of the blank, the standard analytical-chemistry definition.
3. Run a **12-step two-fold serial dilution** from the stock (default 10 mg/mL): 10, 5, 2.5 … 0.0049 mg/mL. Take 3 readings per step and average them.
4. For each Rasa, use its channel:

| Rasa | Channel used |
|---|---|
| Madhura (sweet) | E2 |
| Amla (sour) | pH drop |
| Lavana (salty) | TDS |
| Katu (pungent) | E3 |
| Tikta (bitter) | E1 |
| Kashaya (astringent) | E4 |

5. **Threshold** = the lowest concentration where the signal is still above the LOD, walking down from the strongest concentration until it first drops below.
6. Rasas that aren't characteristic of the herb (reference < 0.1) show "—".

**Chart:** log-log dose-response curves with each signal divided by its own LOD, so the red dashed line at 1.0 is the
detection limit for every channel. A Rasa that is strong in the herb has a **lower threshold**. For Amla, sour and
astringent are detected at more dilute concentrations than sweet.

---

## 9. Database

SQLite file `etongue.db`, a single-file database that needs no setup. The schema is cloud-ready: the same tables
map directly to Firebase / PostgreSQL.

**`samples`**: calibration fingerprints (training data)

| Column | Meaning |
|---|---|
| id, ts | Row id, timestamp |
| label | Herb name (authenticated) |
| source | `reference-lab` (seed data) or `calibration` (added in the app) |
| ph, tds, optical, temp, e1, e2, e3, e4 | Raw sensor values |

**`analyses`**: every test performed

| Column | Meaning |
|---|---|
| id, ts, source | Row id, timestamp, `Simulator` or `ESP32` |
| sample_desc | What was tested (in demo mode, includes the true herb / adulterant) |
| dravya, confidence, quality, status | The AI verdict |
| result_json | Full result (Rasa values, top-3, alerts, distance) |
| reading_json | Averaged sensor reading |

**Continuous learning:** calibrate → samples are added to the `samples` table → **Train Model** → all 6 models are rebuilt from the whole table.

---

## 10. Web dashboard pages

| Page | URL | What you can do | Problem-statement requirement |
|---|---|---|---|
| **Analyze** | `/` | Choose a sample (herb, adulterant, strength, or a 🎭 mystery sample); watch the live sensor values and electrode graph; press **Analyze** to see the herb, confidence, Rasa radar, quality score, closest matches and alerts | E-tongue array, 6 taste modalities, smart alerts and decision support |
| **Calibrate** | `/collect` | Record N readings of an authenticated herb; see how many samples each herb has (herbs with no data are highlighted) | Calibration with authenticated herb samples |
| **Train Model** | `/train` | Rebuild all models from the database; see accuracy, sample count and recognised herbs (your teammate's page, extended) | ML trained on growing datasets |
| **Threshold** | `/threshold` | Run a dilution study for any herb; see the threshold table and dose-response curves | Threshold detection module |
| **Database** | `/database` | Stats, analysis log, **CSV export** of analyses and calibration data, **Reset demo data** | Centralized / cloud-compatible database |

---

## 11. API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/meta` | Herbs, adulterants, data source, calibration counts, model info |
| GET | `/api/reading?herb=&adulterant=&conc=` | One live reading (simulator or latest ESP32) |
| POST | `/api/analyze` `{herb, adulterant, conc}` | Run a full analysis (10 readings) and log it |
| POST | `/api/mystery` | Pick a hidden random sample for mystery mode |
| POST | `/api/collect` `{label, n}` | Add N calibration readings |
| POST | `/make_model` | Retrain all models (the endpoint your teammate's page calls) |
| GET | `/api/threshold?herb=&stock=` | Run a dilution / threshold study |
| GET | `/api/analyses`, `/api/samples` | Database contents as JSON (for cloud sync) |
| GET | `/export/analyses.csv`, `/export/samples.csv` | CSV download |
| POST | `/api/reset` | Restore the starter dataset and model |
| POST | `/api/ingest` | **Hardware input:** the ESP32 posts `{ph, tds, optical, temp, e1, e2, e3, e4}` |

---

## 12. Hardware design (ESP32)

Reference firmware: `firmware/etongue_esp32.ino`. It hasn't been run on a board, because no hardware was available.

| Part | Approx. cost (₹) | Pin |
|---|---|---|
| ESP32 DevKit | 450 | n/a |
| pH sensor module PH-4502C + probe | 900 | GPIO34 |
| TDS sensor module | 400 | GPIO35 |
| 450 nm LED + BPW34 photodiode + LM358 amp + cuvette | 250 | GPIO32 |
| DS18B20 waterproof temperature probe | 150 | GPIO4 |
| 4 × electrodes (carbon / silver wire with PVC-lipid membranes) | 600 | GPIO33, 36, 39, 25 |
| INA333 instrumentation amps / ADS1115 16-bit ADC | 500 | I²C |
| **Total** | **≈ ₹3,000–4,000** | |

**Firmware flow:**
1. Average 20 ADC samples per channel.
2. Convert to units (pH from 2-point calibration, TDS polynomial, absorbance = −log₁₀(V/V_blank), electrode mV).
3. Send JSON over Wi-Fi to `/api/ingest` every 0.5 s.
4. The dashboard automatically switches from the simulator to live ESP32 data.

---

## 13. Demo script for judges

Before you start: **Database → Reset demo data**.

1. **Authentic sample.** On Analyze, choose Tulsi with no adulterant and press Analyze. Expect Tulsi at about 100% confidence, quality 100 and a green "Within threshold" badge. The radar matches the reference profile: Katu (pungent) and Tikta (bitter).
2. **Adulteration.** Keep Tulsi or Neem, set the adulterant to *Synthetic colour* and analyze. Expect a red "Adulteration suspected" badge and the alert "Absorbance higher than reference: added colour".
3. **Haridra + Chalk.** Expect "pH higher than reference: possible alkaline filler".
4. **🎭 Mystery sample.** Let a judge pick it. The AI answers blind, then the true sample is revealed.
5. **Learning a new herb.** Analyze Saindhava and get "Unknown Dravya: not in library". On Calibrate, record 40 Saindhava readings. On Train Model, the recognised herbs go from 7 to 8. Analyze Saindhava again and it is correctly identified.
6. **Threshold.** Choose Amla and run the series to show the curves and the threshold table.
7. **Database.** Show the full log and download the CSV.

---

## 14. Questions judges may ask

**"How does it know the adulterant without tasting it?"**
The sensors do the "tasting". The system learned what authentic Neem looks like across 8 sensors. If a sample falls
outside that range, it's flagged, and this needs no knowledge of the adulterant. The pattern of which sensors changed
then points to the likely cause. For example, higher absorbance suggests dye, and higher pH suggests chalk. It's an
evidence-based hypothesis; a lab test (HPTLC / LC-MS) confirms it. With real hardware we would also record known
adulterated mixtures (e.g. Neem + 5% chalk) so the model learns each adulterant's real fingerprint.

**"Is the data real?"**
No. The sensor readings are simulated because we had no hardware. The ML, alerts, database and threshold logic are
real, and would run unchanged on ESP32 data through `/api/ingest`.

**"Why not just use HPLC?"**
HPLC costs lakhs, needs a lab and a trained operator, and takes hours. This device costs a few thousand rupees, is
portable and answers in seconds, so it works as a **screening** tool at the point of purchase. HPLC stays for confirmation.

**"What if it sees a herb it was never trained on?"**
It reports "Unknown Dravya" instead of guessing (demo step 5).

**"Why Random Forest and not deep learning?"**
We have hundreds of samples, not millions. Random Forest is more accurate on small tabular data, gives probabilities,
is explainable, and trains in under a second. A neural network becomes worthwhile once the database grows large.

**"How does it handle batch-to-batch variation?"**
Calibration includes samples from 0.85× to 1.15× strength, and the acceptance limit is learned from that real spread.

---

## 15. Limitations (be honest about these)

- All sensor data is **simulated**; accuracy on real herbs is unproven.
- Adulterant effects in the simulator were **designed by us**, so detecting them is partly circular. Real-world performance needs real adulterated samples.
- Rasa targets come from classical texts; real Rasa calibration would need trained human taste panels.
- Under heavy filler (starch, chalk) the herb identity can be wrong (~14% in tests), although the sample is still flagged as adulterated.
- The thresholds and limits (99th percentile × 1.15, unknown factor 15, confidence 60%) were tuned on simulated data.
- The firmware is a reference design that hasn't been tested on hardware.

---

## 16. Roadmap

1. **Build the hardware** (section 12) and replace the simulator with real readings.
2. **Real calibration library:** pharmacopoeia-authenticated samples from multiple suppliers and seasons.
3. **Supervised adulterant classifier:** record known adulterated mixtures and train a model to name the adulterant directly.
4. **Phytochemical correlation:** join HPTLC / FTIR / LC-MS results per batch, and train a "sensor fingerprint → marker compound" model (e.g. curcumin %, glycyrrhizin %).
5. **Spectroscopy add-on:** NIR or UV-Vis spectra as extra features.
6. **Cloud sync:** push the `analyses` / `samples` tables to Firebase so all devices share one learning database.
7. **Edge AI:** export the trained model to run on the ESP32 itself (e.g. TensorFlow Lite Micro or a decision-tree port) for offline field use.
8. **Mobile app** for field inspectors; QR-coded batch reports for regulators.

---

## 17. Tech stack and file structure

| Layer | Technology |
|---|---|
| Hardware (planned) | ESP32, pH, TDS, optical, DS18B20, 4-electrode array |
| Backend | Python 3.12, Flask |
| ML | scikit-learn (Random Forest, Ridge, StandardScaler), NumPy (Mahalanobis, LOD) |
| Database | SQLite (cloud-ready schema, JSON/CSV export) |
| Frontend | HTML, CSS, JavaScript, Chart.js 4.4.1 (bundled locally) |

```
Try/
├── app.py                 Flask server: pages, API, SQLite database, seeding
├── ml.py                  All models: classifier, Rasa regressor, anomaly detector, explanations, threshold study
├── sim.py                 Simulated sensor array: herb Rasa profiles, adulterants, noise
├── requirements.txt       flask, scikit-learn, numpy
├── templates/
│   ├── base.html          Shared layout + navigation
│   ├── index.html         Analyze page (main demo)
│   ├── collect.html       Calibrate page
│   ├── train.html         Train Model page (based on teammate's train.html)
│   ├── threshold.html     Threshold detection page
│   └── database.html      Database / export page
├── static/
│   ├── style.css          Theme (teammate's dark-green design)
│   └── chart.umd.min.js   Chart.js (offline)
├── firmware/
│   └── etongue_esp32.ino  ESP32 reference firmware
├── train.html             Teammate's original page (kept for reference)
├── etongue.db             Database (created on first run)
└── model.pkl              Trained models (created on first run)
```
