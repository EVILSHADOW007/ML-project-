# Urban Air Quality Profiling (B.Tech ML Subject Project)

An interactive, instrumentation-style web application for showcasing the 3rd-year Machine Learning subject project:
**"Urban Air Quality Profiling using Imputation, Covariance, Polynomial Curve Fitting, Ridge Regression, and SVM Classification"**.

Designed following the **shadcn/ui** modern zinc dark theme (`#09090b`), hairline borders (`#27272a`), live animated AQI gauge dial, and JetBrains Mono data typography.

---

## 📊 Dataset Specification

All models, statistical analyses, regressions, and classifications are derived **exclusively** from the uploaded UCI Air Quality dataset:
- **File**: `data/AirQualityUCI.csv` (semicolon-delimited, comma decimal separator, `-200` missing-value sentinel)
- **Timeframe**: March 10, 2004 – April 4, 2005 (1 year of continuous hourly atmospheric monitoring in an Italian city)
- **Instances**: 9,357 valid hourly observations
- **Target Pollutants & Chemical Sensors**:
  - `CO(GT)`: Reference Carbon Monoxide concentration in mg/m³
  - `PT08.S1(CO)`: Tin oxide sensor response (targeted for CO)
  - `NMHC(GT)`: Non-Methane Hydrocarbons in µg/m³
  - `C6H6(GT)`: Benzene concentration in µg/m³
  - `PT08.S2(NMHC)`: Titania sensor response (targeted for NMHC)
  - `NOx(GT)`: Nitrogen Oxides in ppb
  - `PT08.S3(NOx)`: Tungsten oxide sensor response (targeted for NOx)
  - `NO2(GT)`: Nitrogen Dioxide in µg/m³
  - `PT08.S4(NO2)`: Tungsten oxide sensor response (targeted for NO2)
  - `PT08.S5(O3)`: Indium oxide sensor response (targeted for O3)
  - `T`: Ambient Temperature (°C)
  - `RH`: Relative Humidity (%)
  - `AH`: Absolute Humidity

---

## 📁 Repository Structure

```
/
├── index.html                   -> Landing Page: Animated AQI dial hero, 5-unit syllabus grid
├── unit1/
│   ├── index.html               -> Unit 1 Overview: 3 algorithm cards with live dataset sparklines
│   ├── imputation.html          -> Mean / Median / KNN (k=5) toggle, missing point pulse & staggered fill animation
│   ├── covariance.html          -> Interactive 6×6 correlation heatmap, locked scatter detail, top-3 correlated pairs
│   └── polynomial.html          -> Degree 1-4 slider, smooth curve morphing, R²/RMSE count-ups, dynamic copy
├── unit2/
│   ├── index.html               -> Unit 2 Overview: Ridge Regression & SVM Classification cards
│   ├── ridge.html               -> Multicollinearity diagnostic, α ∈ [0, 100] penalty shrinkage & sign-flip mitigation
│   └── svm.html                 -> Safe/Moderate/Hazardous classification, Linear vs RBF, 3×3 confusion matrix & 2D decision boundary
├── data/
│   ├── AirQualityUCI.csv        -> Raw UCI Air Quality dataset (semicolon-delimited)
│   ├── overview_results.json    -> Dataset summary + EPA CO air-quality index sweep driving the AQI dial
│   ├── imputation_results.json  -> Missing telemetry & imputed values for Mean, Median, and KNN
│   ├── covariance_results.json  -> 6×6 correlation matrix and full pairwise scatter data
│   ├── polyfit_results.json     -> Degrees 1-4 curve data, equations, R², RMSE, and overfitting analysis
│   ├── ridge_results.json       -> OLS vs Ridge models across α penalties, coefficient shrinkage & scatter samples
│   └── svm_results.json         -> Linear and RBF classification metrics, confusion matrices & 2D boundary grid
├── scripts/
│   └── generate_all_results.py  -> Python pipeline script to recompute all JSON metrics from AirQualityUCI.csv
├── tests/
│   └── verify_graph_data.mjs    -> Re-parses the raw CSV and asserts every chart's input matches it
├── assets/
│   ├── style.css                -> shadcn/ui zinc dark design system, pill badges, hairline panels, AQI scales
│   ├── gauge.js                 -> Reusable Canvas AQI gauge engine (used in hero & persistent nav chip)
│   └── charts.js                -> Chart.js setup, point tweening, curve morphing, and interactive managers
└── README.md                    -> Project documentation & data integration guide
```

---

## 🚀 How to Serve / Run Locally

This is a pure static web application (HTML5, CSS3, Vanilla JS, Chart.js via CDN). No node modules required!

### Option 1: Python HTTP Server (Recommended)
Open your terminal in the project directory:
```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000` in your web browser.

### Option 2: Live Server Extension
Open the workspace in VS Code, right-click `index.html` and select **Open with Live Server**.

### Option 3: Recompute Dataset Results
To re-run all calculations from `data/AirQualityUCI.csv`:
```bash
python scripts/generate_all_results.py
```
Requires `pandas`, `numpy` and `scikit-learn`.

### Option 4: Verify the Charts Against the Dataset
`tests/verify_graph_data.mjs` re-parses `data/AirQualityUCI.csv` with its own
independent parser, imports the real chart-data-preparation functions out of
`assets/charts.js`, and asserts that the numbers reaching every chart match the
CSV — point values, axis labels, sample sizes, missing-value flags and class
labels included. No dependencies beyond Node.

```bash
node tests/verify_graph_data.mjs
```

---

## 📈 How the Numbers Reach the Screen

Every figure on the site is precomputed by `scripts/generate_all_results.py` into
the JSON files under `data/`, then read by `assets/charts.js` and `assets/gauge.js`
at page load. **No dataset value is hard-coded in the HTML, CSS or JS** — the
static text in the markup is only a `--` / "loading…" placeholder that the JSON
replaces.

Notable derivations:

- **Correlation matrix** uses pairwise-complete rows, so each pair reports its own
  sample size (`pair_counts` in `covariance_results.json`) rather than one
  all-column complete-case count.
- **Imputation MAE** for Mean, Median and KNN is scored over the *same* randomly
  masked cells of the *same* fully-observed rows, so the three tabs are directly
  comparable. The panel shows the MAE of the plotted column, `CO(GT)`, never the
  cross-column average (which mixes mg/m³, ppb and µg/m³).
- **Ridge and SVM** fit `StandardScaler` on the training split only, so the
  reported R²/accuracy are not inflated by test-set leakage.
- **SVM decision boundary** is a 2-feature model over the raw channels
  `PT08.S1(CO)` and `PT08.S2(NMHC)` — one grid per kernel, so the regions always
  match the selected kernel. It is *not* PCA.
- **AQI dial** steps through real consecutive hourly readings: the US EPA
  carbon-monoxide index of the 8-hour running mean of `CO(GT)`
  (`ppm = mg/m³ × 24.45 / 28.01`). The whole derivation is exact integer
  arithmetic so the published index is reproducible. Over this dataset the index
  spans AQI 2–79 (mean 21.7). If `overview_results.json` cannot be loaded the dial
  reads `--` rather than showing a placeholder value.

---

## 🎨 Design System & Color Standards (shadcn/ui Zinc Dark)

- **Background**: Deep Zinc Dark `#09090b` with subtle ambient radial glow
- **Cards & Panels**: Surface Zinc `#111114` with `#18181b` hover
- **Borders**: Hairline Zinc `#27272a`
- **Text**: Off-White `#fafafa` (Primary Foreground), Neutral Zinc `#a1a1aa` (Muted Foreground)
- **Buttons**:
  - Primary Action: Signature shadcn white button `#fafafa` with dark text `#09090b`
  - Secondary Action: Zinc button `#18181b` with border `#27272a`
  - Segmented Tabs: Container `#18181b` with active pill `#09090b`
- **AQI Color Scale**:
  - Good (0-50): `#34d399` (Emerald-400)
  - Moderate (51-100): `#fbbf24` (Amber-400)
  - Sensitive (101-150): `#fb923c` (Orange-400)
  - Unhealthy (151-200): `#f87171` (Rose-400)
  - Hazardous (201+): `#c084fc` (Purple-400)
- **Typography**: Inter (Headings/UI), JetBrains Mono (Numbers/Metrics/Equations/Data).
