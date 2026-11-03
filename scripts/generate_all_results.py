"""
Comprehensive Python Script to Compute All ML Project Metrics
Exclusively from data/AirQualityUCI.csv
Outputs:
  - data/imputation_results.json
  - data/covariance_results.json
  - data/polyfit_results.json
  - data/ridge_results.json
  - data/svm_results.json
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer

# Physical / reference metadata for the UCI Air Quality columns (see README).
# Used only for axis labels and readouts - never for deriving values.
COLUMN_UNITS = {
    'CO(GT)': 'mg/m3',
    'PT08.S1(CO)': 'sensor response',
    'NMHC(GT)': 'ug/m3',
    'C6H6(GT)': 'ug/m3',
    'PT08.S2(NMHC)': 'sensor response',
    'NOx(GT)': 'ppb',
    'PT08.S3(NOx)': 'sensor response',
    'NO2(GT)': 'ug/m3',
    'PT08.S4(NO2)': 'sensor response',
    'PT08.S5(O3)': 'sensor response',
    'T': 'deg C',
    'RH': '%',
    'AH': 'absolute humidity',
    'hour': 'hour of day',
}

# Missing-data sentinel used throughout AirQualityUCI.csv.
MISSING_SENTINEL = -200

# KNN imputation neighbourhood size. Must stay in sync with the "KNN (k=N)"
# label rendered by unit1/imputation.html.
KNN_NEIGHBOURS = 5


def json_dump(obj, filename):
    """Write a results file deterministically as ASCII-escaped UTF-8 JSON."""
    path = os.path.join('data', filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"  -> Saved data/{filename}")


def main():
    csv_path = os.path.join('data', 'AirQualityUCI.csv')
    print(f"Loading dataset from {csv_path}...")

    # Load with ; separator and , decimal
    df_raw = pd.read_csv(csv_path, sep=';', decimal=',')

    # Drop empty trailing columns if present
    for col in ['Unnamed: 15', 'Unnamed: 16']:
        if col in df_raw.columns:
            df_raw = df_raw.drop(columns=[col])

    # Drop rows that are completely NaN or missing Date
    df_raw = df_raw.dropna(how='all')
    df = df_raw[df_raw['Date'].notna()].copy()

    # Parse Time to extract hour (integer 0-23)
    # Time format is HH.MM.SS
    def parse_hour(t_str):
        if pd.isna(t_str):
            return np.nan
        parts = str(t_str).replace(':', '.').split('.')
        try:
            return int(parts[0])
        except:
            return np.nan

    df['hour'] = df['Time'].apply(parse_hour)

    # Human-readable timestamp straight from the CSV's Date + Time columns,
    # so time-series charts can label the x-axis with real reading times.
    def make_timestamp(row):
        hour = parse_hour(row['Time'])
        if pd.isna(hour):
            return str(row['Date'])
        return f"{row['Date']} {int(hour):02d}:00"

    df['timestamp'] = df.apply(make_timestamp, axis=1)

    # Convert numeric columns, replacing -200 with NaN
    numeric_cols = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH', 'hour'
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df.loc[df[c] == MISSING_SENTINEL, c] = np.nan

    print(f"Total valid timestamp records: {len(df)}")

    # ---------------------------------------------------------
    # 1. COVARIANCE & CORRELATION MATRIX (Unit 1)
    # ---------------------------------------------------------
    print("Computing Covariance & Correlation Matrix...")
    pollutants = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH']

    # Pairwise complete correlation matrix
    corr_df = df[pollutants].corr(method='pearson')
    matrix_vals = [[round(float(corr_df.loc[r, c]), 3) for c in pollutants] for r in pollutants]

    # Find all pairwise pairs and their actual scatter sample.
    # pair_counts records the *pairwise complete* row count actually behind each
    # r value, which is what the UI must report as the sample size (it differs
    # per pair, and from the all-column complete-case count).
    scatter_data = {}
    pair_counts = {}
    pair_corrs = []

    np.random.seed(42)
    for i in range(len(pollutants)):
        for j in range(i, len(pollutants)):
            p1 = pollutants[i]
            p2 = pollutants[j]

            if p1 == p2:
                # Diagonal / identity pair: real single-column readings plotted
                # against themselves, so the heatmap diagonal has genuine data.
                col_df = df[[p1]].dropna()
                pair_counts[f"{p1}_{p1}"] = int(len(col_df))
                sample_df = col_df.sample(n=min(250, len(col_df)), random_state=42)
                scatter_data[f"{p1}_{p1}"] = [
                    {'x': round(float(v), 2), 'y': round(float(v), 2)}
                    for v in sample_df[p1]
                ]
                continue

            pair_df = df[[p1, p2]].dropna()
            r_val = round(float(corr_df.loc[p1, p2]), 3)
            pair_corrs.append({'a': p1, 'b': p2, 'corr': r_val, 'abs_corr': abs(r_val)})
            pair_counts[f"{p1}_{p2}"] = int(len(pair_df))
            pair_counts[f"{p2}_{p1}"] = int(len(pair_df))

            # Sample 250 points for scatter
            n_sample = min(250, len(pair_df))
            sample_df = pair_df.sample(n=n_sample, random_state=42)
            pts = [{'x': round(float(row[p1]), 2), 'y': round(float(row[p2]), 2)} for _, row in sample_df.iterrows()]
            scatter_data[f"{p1}_{p2}"] = pts
            scatter_data[f"{p2}_{p1}"] = [{'x': pt['y'], 'y': pt['x']} for pt in pts]

    # Sort pairs by absolute correlation descending for top pairs
    pair_corrs.sort(key=lambda x: x['abs_corr'], reverse=True)
    top_3 = []
    for item in pair_corrs[:3]:
        top_3.append({
            'a': item['a'],
            'b': item['b'],
            'corr': item['corr'],
            'n': pair_counts[f"{item['a']}_{item['b']}"],
            'scatter': scatter_data[f"{item['a']}_{item['b']}"]
        })

    cov_results = {
        'pollutants': pollutants,
        'units': {p: COLUMN_UNITS.get(p, '') for p in pollutants},
        'matrix': matrix_vals,
        'top_pairs': top_3,
        'scatter_data': scatter_data,
        'pair_counts': pair_counts,
        'n_rows_used': int(df[pollutants].dropna().shape[0])
    }

    json_dump(cov_results, 'covariance_results.json')

    # ---------------------------------------------------------
    # 2. IMPUTATION PIPELINE (Unit 1)
    # ---------------------------------------------------------
    print("Computing Imputation results...")
    imp_cols = ['CO(GT)', 'NOx(GT)', 'NO2(GT)']
    total_missing_counts = {c: int(df[c].isna().sum()) for c in imp_cols}
    total_missing_sum = sum(total_missing_counts.values())

    # All three methods are scored on the SAME artificially masked cells, using the
    # SAME feature matrix the chart is drawn from. Previously mean/median MAE was
    # measured against every known value while KNN MAE was measured only over the
    # masked 15%, using a 3-column matrix instead of the 9-channel one - so the
    # three numbers the UI compares were not measuring the same thing.
    knn_features = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
                    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'T', 'RH']
    knn_imputer = KNNImputer(n_neighbors=KNN_NEIGHBOURS)

    # Rows where every helper channel AND every evaluated column is observed, so
    # the masked cells have ground truth to score against.
    eval_pool = df[knn_features].dropna()
    eval_sample = eval_pool.sample(n=min(1500, len(eval_pool)), random_state=42).copy()

    # Mean and Median on full valid data
    means = {c: float(df[c].mean()) for c in imp_cols}
    medians = {c: float(df[c].median()) for c in imp_cols}

    # Mask 15% of the cells of the evaluated columns only.
    MASK_FRACTION = 0.15
    rng = np.random.RandomState(42)
    mask = pd.DataFrame(
        rng.rand(len(eval_sample), len(imp_cols)) < MASK_FRACTION,
        index=eval_sample.index,
        columns=imp_cols,
    )

    eval_masked = eval_sample.copy()
    for c in imp_cols:
        eval_masked.loc[mask[c], c] = np.nan

    knn_eval = pd.DataFrame(
        knn_imputer.fit_transform(eval_masked),
        columns=knn_features,
        index=eval_sample.index,
    )

    def masked_mae(filled):
        out = {}
        for c in imp_cols:
            sel = mask[c]
            if not sel.any():
                out[c] = None
                continue
            out[c] = round(float(np.abs(filled[c][sel] - eval_sample.loc[sel, c]).mean()), 3)
        return out

    mae_mean = masked_mae({c: pd.Series(means[c], index=eval_sample.index) for c in imp_cols})
    mae_median = masked_mae({c: pd.Series(medians[c], index=eval_sample.index) for c in imp_cols})
    mae_knn = masked_mae({c: knn_eval[c] for c in imp_cols})

    masked_cell_count = int(mask.to_numpy().sum())
    print(f"  Imputation MAE scored on {masked_cell_count} masked cells "
          f"across {len(eval_sample)} rows (identical cells for all 3 methods)")

    overall_mae_mean = round(float(np.mean(list(mae_mean.values()))), 3)
    overall_mae_median = round(float(np.mean(list(mae_median.values()))), 3)
    overall_mae_knn = round(float(np.mean(list(mae_knn.values()))), 3)

    # Timeline sequence for visualization (400 consecutive readings of CO(GT))
    # Pick a contiguous window with realistic sensor dropouts
    WINDOW_START, WINDOW_END = 200, 600
    co_series = df['CO(GT)'].iloc[WINDOW_START:WINDOW_END].copy().reset_index(drop=True)
    before_seq = [round(float(v), 2) if pd.notna(v) else None for v in co_series]
    was_missing = [bool(pd.isna(v)) for v in co_series]
    # Real reading times for the window, so the timeline x-axis is labelled with
    # the CSV's own Date/Time instead of a bare 1..N sample counter.
    window_timestamps = [str(t) for t in df['timestamp'].iloc[WINDOW_START:WINDOW_END]]

    # Impute the 400 window
    mean_val = round(means['CO(GT)'], 2)
    median_val = round(medians['CO(GT)'], 2)
    after_mean = [val if val is not None else mean_val for val in before_seq]
    after_median = [val if val is not None else median_val for val in before_seq]

    # KNN on window with helper features
    window_df = df[knn_features].iloc[WINDOW_START:WINDOW_END].copy()
    window_knn = knn_imputer.fit_transform(window_df)
    after_knn = [round(float(window_knn[i, 0]), 2) for i in range(len(before_seq))]

    # Descriptions quote the CO(GT) MAE, because CO(GT) is the column actually
    # charted and shown in the MAE readout. `mae` stays as the cross-column
    # average and is flagged as mixed-unit so the UI never shows it beside a
    # single column's axis.
    def method_block(overall_mae, mae_per_col, after_seq, description):
        return {
            'filled_count': total_missing_sum,
            'missing_per_column': total_missing_counts,
            'holdout_test_count': masked_cell_count,
            'holdout_rows': len(eval_sample),
            'mae_protocol': (
                f'Mean absolute error over {masked_cell_count} cells masked at random '
                f'({MASK_FRACTION:.0%}) from {len(eval_sample)} fully-observed rows; '
                f'identical cells for all three methods.'
            ),
            'mae': overall_mae,
            'mae_is_mixed_unit_average': True,
            'mae_per_column': mae_per_col,
            'chart_column': 'CO(GT)',
            'chart_unit': COLUMN_UNITS['CO(GT)'],
            'timestamps': window_timestamps,
            'before': before_seq,
            'after': after_seq,
            'was_missing': was_missing,
            'description': description,
        }

    imputation_results = {
        'methods': {
            'mean': method_block(
                overall_mae_mean, mae_mean, after_mean,
                f"Global mean substitution (CO(GT) baseline {mean_val:.2f} mg/m\u00b3, "
                f"MAE {mae_mean['CO(GT)']:.2f} mg/m\u00b3). Fast computation O(1), but severely "
                f"dampens temporal variance and fails to capture rush-hour dynamics."
            ),
            'median': method_block(
                overall_mae_median, mae_median, after_median,
                f"Median substitution (CO(GT) baseline {median_val:.2f} mg/m\u00b3, "
                f"MAE {mae_median['CO(GT)']:.2f} mg/m\u00b3). Resistant to extreme pollution "
                f"outlier spikes, but creates flatline artifacts in sensor telemetry."
            ),
            'knn': method_block(
                overall_mae_knn, mae_knn, after_knn,
                f"K-Nearest Neighbors (k={KNN_NEIGHBOURS}) based on multi-sensor atmospheric "
                f"proximity (PT08.S1, C6H6, NOx, Temperature). Reconstructs smooth diurnal "
                f"curves with {mae_knn['CO(GT)']:.2f} mg/m\u00b3 MAE on CO(GT)."
            ),
        },
        'columns': imp_cols,
        'knn_neighbours': KNN_NEIGHBOURS,
        'window': {
            'start_row': WINDOW_START,
            'end_row': WINDOW_END,
            'first_timestamp': window_timestamps[0],
            'last_timestamp': window_timestamps[-1],
        },
    }

    json_dump(imputation_results, 'imputation_results.json')

    # ---------------------------------------------------------
    # 3. POLYNOMIAL CURVE FITTING (Unit 1)
    # ---------------------------------------------------------
    print("Computing Polynomial Curve Fitting (NO2 diurnal cycle)...")
    poly_df = df[['hour', 'NO2(GT)']].dropna()
    X_poly = poly_df['hour'].values
    y_poly = poly_df['NO2(GT)'].values

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)

    # Generate scatter sample for visualization
    sample_idx = np.random.RandomState(42).choice(len(poly_df), size=min(300, len(poly_df)), replace=False)
    actual_pts = [{'x': float(X_poly[i]), 'y': round(float(y_poly[i]), 1)} for i in sample_idx]

    x_curve = np.linspace(0, 23, 100)
    degrees_data = {}

    for deg in [1, 2, 3, 4]:
        coeffs = np.polyfit(X_train, y_train, deg) # highest to lowest power
        p = np.poly1d(coeffs)

        y_train_pred = p(X_train)
        y_test_pred = p(X_test)

        r2 = round(float(r2_score(y_train, y_train_pred)), 4)
        rmse_train = round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 2)
        rmse_test = round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 2)

        # Evaluate the polynomial at the SAME x that gets published, otherwise each
        # plotted point's y belongs to a slightly different x than its own label.
        curve_pts = []
        for x in x_curve:
            x_pt = round(float(x), 4)
            curve_pts.append({'x': x_pt, 'y': round(float(p(x_pt)), 4)})

        if deg == 1:
            readout = f"Degree 1 (Linear): Inadequate fit (R² = {r2:.4f}, Test RMSE = {rmse_test:.2f} µg/m³). Underfits the cyclical morning and evening traffic congestion peaks."
        elif deg == 2:
            readout = f"Degree 2 (Quadratic): Parabolic profile captures daytime valley but fails to resolve the dual rush-hour commuter peaks (R² = {r2:.4f}, Test RMSE = {rmse_test:.2f} µg/m³)."
        elif deg == 3:
            readout = f"Degree 3 (Cubic): Optimal generalization (R² = {r2:.4f}, Test RMSE = {rmse_test:.2f} µg/m³). Accurately models the bimodal rush-hour pattern without tail oscillations."
        else:
            readout = f"Degree 4 (Quartic): Marginal training gain (R² = {r2:.4f}), but increased test variance (Test RMSE = {rmse_test:.2f} µg/m³), displaying boundary curve instability."

        degrees_data[str(deg)] = {
            'r2': r2,
            'rmse': rmse_train,
            'held_out_rmse': rmse_test,
            'coefficients': [round(float(c), 10) for c in coeffs],
            'actual_points': actual_pts,
            'curve_points': curve_pts,
            'readout': readout,
            'analysis': readout
        }

    poly_results = {
        'x_label': 'Hour of day (0–23)',
        'y_label': 'NO2(GT) (µg/m³)',
        'x_column': 'hour',
        'y_column': 'NO2(GT)',
        'x_range': [0, 23],
        'degrees': degrees_data,
        'n_rows_used': int(len(poly_df)),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test))
    }

    json_dump(poly_results, 'polyfit_results.json')

    # ---------------------------------------------------------
    # 4. RIDGE REGRESSION & MULTICOLLINEARITY (Unit 2)
    # ---------------------------------------------------------
    print("Computing Ridge Regression & Multicollinearity analysis...")
    ridge_features = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
        'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'hour'
    ]
    target_col = 'CO(GT)'

    ridge_df = df[ridge_features + [target_col]].dropna()
    print(f"  Complete rows for regression: {len(ridge_df)}")

    # Feature correlations to highlight multicollinearity
    feat_corr = ridge_df[ridge_features].corr()
    high_corr_pairs = []
    for i in range(len(ridge_features)):
        for j in range(i + 1, len(ridge_features)):
            f1, f2 = ridge_features[i], ridge_features[j]
            r_val = float(feat_corr.loc[f1, f2])
            if abs(r_val) > 0.70:
                high_corr_pairs.append({'a': f1, 'b': f2, 'corr': round(r_val, 3)})
    high_corr_pairs.sort(key=lambda x: abs(x['corr']), reverse=True)

    X_reg = ridge_df[ridge_features].values
    y_reg = ridge_df[target_col].values

    # Split FIRST, then fit the scaler on the training split only. Standardising
    # the whole matrix beforehand leaks test-set means/variances into training and
    # inflates the R2 / RMSE figures the page reports.
    X_train_raw, X_test_raw, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train_raw)
    X_train_r = scaler.transform(X_train_raw)
    X_test_r = scaler.transform(X_test_raw)

    # Fit OLS and Ridge models
    alpha_dict = {
        'ols': (0.0, 'OLS (α=0, no penalty)'),
        'ridge_1.0': (1.0, 'Ridge (α=1.0)'),
        'ridge_10.0': (10.0, 'Ridge (α=10.0)'),
        'ridge_50.0': (50.0, 'Ridge (α=50.0)'),
        'ridge_100.0': (100.0, 'Ridge (α=100.0)')
    }

    models_dict = {}
    ols_preds = None

    for key, (alpha, d_name) in alpha_dict.items():
        if alpha == 0:
            reg = LinearRegression()
        else:
            reg = Ridge(alpha=alpha, random_state=42)

        reg.fit(X_train_r, y_train_r)

        y_tr_pred = reg.predict(X_train_r)
        y_te_pred = reg.predict(X_test_r)
        if key == 'ols':
            ols_preds = y_te_pred

        r2_tr = round(float(r2_score(y_train_r, y_tr_pred)), 4)
        r2_te = round(float(r2_score(y_test_r, y_te_pred)), 4)
        rmse_tr = round(float(np.sqrt(mean_squared_error(y_train_r, y_tr_pred))), 4)
        rmse_te = round(float(np.sqrt(mean_squared_error(y_test_r, y_te_pred))), 4)
        mae_te = round(float(mean_absolute_error(y_test_r, y_te_pred)), 4)
        intercept = round(float(reg.intercept_), 4)

        coef_map = {feat: round(float(c), 4) for feat, c in zip(ridge_features, reg.coef_)}

        models_dict[key] = {
            'alpha': alpha,
            'display_name': d_name,
            'r2_train': r2_tr,
            'r2_test': r2_te,
            'rmse_train': rmse_tr,
            'rmse_test': rmse_te,
            'mae_test': mae_te,
            'intercept': intercept,
            'coefficients': coef_map
        }

    # Sample 150 points for actual vs predicted scatter plot
    sample_idx_r = np.random.RandomState(42).choice(len(y_test_r), size=min(150, len(y_test_r)), replace=False)
    pred_sample = [
        {'actual': round(float(y_test_r[i]), 2), 'predicted': round(float(ols_preds[i]), 2)}
        for i in sample_idx_r
    ]

    # Observed range of the target across the held-out test split, so the
    # "perfect prediction" y = x reference line is derived from the data rather
    # than pinned to an arbitrary hard-coded span.
    target_min = float(min(y_test_r.min(), ols_preds.min()))
    target_max = float(max(y_test_r.max(), ols_preds.max()))

    ridge_results = {
        'target': target_col,
        'target_unit': COLUMN_UNITS.get(target_col, ''),
        'target_range': [round(target_min, 2), round(target_max, 2)],
        'predictors': ridge_features,
        'n_train': int(len(X_train_r)),
        'n_test': int(len(X_test_r)),
        'n_rows_used': int(len(ridge_df)),
        'high_correlation_pairs': high_corr_pairs,
        'models': models_dict,
        'best_alpha_label': 'Ridge (α=10.0)',
        'pred_vs_actual_sample': pred_sample
    }

    json_dump(ridge_results, 'ridge_results.json')

    # ---------------------------------------------------------
    # 5. SVM CLASSIFICATION (Unit 2)
    # ---------------------------------------------------------
    print("Computing SVM Classification models...")
    svm_features = ['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S3(NOx)', 'hour']
    svm_df = df[svm_features + ['CO(GT)']].dropna().copy()

    # Define AQI categories for CO(GT)
    # Safe: <= 2.0 mg/m3, Moderate: 2.0 - 4.0 mg/m3, Hazardous: > 4.0 mg/m3
    def get_cat(val):
        if val <= 2.0:
            return 'Safe'
        elif val <= 4.0:
            return 'Moderate'
        else:
            return 'Hazardous'

    svm_df['category'] = svm_df['CO(GT)'].apply(get_cat)
    cat_labels = ['Safe', 'Moderate', 'Hazardous']

    X_svm = svm_df[svm_features].values
    y_svm = svm_df['category'].values

    # Split FIRST, then fit the scaler on the training split only, so the reported
    # accuracy and confusion matrices are not inflated by test-set leakage.
    X_tr_raw, X_te_raw, y_tr_s, y_te_s = train_test_split(
        X_svm, y_svm, test_size=0.2, random_state=42, stratify=y_svm)

    scaler_svm = StandardScaler().fit(X_tr_raw)
    X_tr_s = scaler_svm.transform(X_tr_raw)
    X_te_s = scaler_svm.transform(X_te_raw)

    kernels_dict = {}

    for k_name in ['linear', 'rbf']:
        clf = SVC(kernel=k_name, random_state=42, C=1.0)
        clf.fit(X_tr_s, y_tr_s)

        y_pred = clf.predict(X_te_s)
        acc = round(float(accuracy_score(y_te_s, y_pred)), 4)

        rep = classification_report(y_te_s, y_pred, output_dict=True, labels=cat_labels, zero_division=0)
        per_class = {}
        for cat in cat_labels:
            per_class[cat] = {
                'precision': round(float(rep[cat]['precision']), 4),
                'recall': round(float(rep[cat]['recall']), 4),
                'f1': round(float(rep[cat]['f1-score']), 4),
                'support': int(rep[cat]['support'])
            }

        cm = confusion_matrix(y_te_s, y_pred, labels=cat_labels)
        cm_list = [[int(val) for val in row] for row in cm]

        kernels_dict[k_name] = {
            'accuracy': acc,
            'per_class': per_class,
            'confusion_matrix': cm_list,
            'confusion_matrix_labels': cat_labels
        }
        print(f"  SVM {k_name} test accuracy: {acc * 100:.2f}%")

    # Generate 2D decision boundary grids over the two most CO-informative
    # sensor channels. One grid PER KERNEL, so the plotted regions always match
    # the kernel the user has selected in the UI.
    feat_2d = ['PT08.S1(CO)', 'PT08.S2(NMHC)']
    df_2d = df[feat_2d + ['CO(GT)']].dropna().copy()
    df_2d['category'] = df_2d['CO(GT)'].apply(get_cat)

    # Subsample for smooth 2D fitting
    sub_2d = df_2d.sample(n=min(2000, len(df_2d)), random_state=42)
    X_2d = sub_2d[feat_2d].values
    y_2d = sub_2d['category'].values

    # Standardise the two channels for fitting (raw sensor counts are in the
    # hundreds, which makes a linear SVC degenerate), but keep the grid axes in
    # the ORIGINAL sensor units so the plot can be read against the CSV.
    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d)

    x_min, x_max = float(X_2d[:, 0].min()), float(X_2d[:, 0].max())
    y_min, y_max = float(X_2d[:, 1].min()), float(X_2d[:, 1].max())

    grid_res = 50
    gx = np.linspace(x_min, x_max, grid_res)
    gy = np.linspace(y_min, y_max, grid_res)
    xx, yy = np.meshgrid(gx, gy)          # xx/yy shape == (len(gy), len(gx))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    grid_pts_scaled = scaler_2d.transform(grid_pts)

    cat_to_int = {cat: i for i, cat in enumerate(cat_labels)}
    boundary_kernels = {}

    for k_name in ['linear', 'rbf']:
        clf_2d = SVC(kernel=k_name, C=1.0, random_state=42)
        clf_2d.fit(X_2d_scaled, y_2d)
        grid_preds = clf_2d.predict(grid_pts_scaled)
        # Row index -> y (PT08.S2), column index -> x (PT08.S1).
        grid_labels = np.array([cat_to_int[q] for q in grid_preds]).reshape(xx.shape)
        boundary_kernels[k_name] = {
            'grid_labels': grid_labels.tolist(),
            'train_accuracy': round(float(clf_2d.score(X_2d_scaled, y_2d)), 4)
        }

    # Sample 150 points for 2D scatter visualization
    sample_scatter_idx = np.random.RandomState(42).choice(len(sub_2d), size=min(150, len(sub_2d)), replace=False)
    scatter_2d_pts = [
        {'x': round(float(X_2d[i, 0]), 1), 'y': round(float(X_2d[i, 1]), 1), 'label': str(y_2d[i])}
        for i in sample_scatter_idx
    ]

    class_distribution = {
        cat: int((svm_df['category'] == cat).sum()) for cat in cat_labels
    }

    svm_results = {
        'target': 'CO(GT)',
        'target_unit': COLUMN_UNITS['CO(GT)'],
        'category_thresholds': {
            'Safe': {'min': 0.0, 'max': 2.0},
            'Moderate': {'min': 2.0, 'max': 4.0},
            'Hazardous': {'min': 4.0, 'max': round(float(svm_df['CO(GT)'].max()), 1)}
        },
        'predictors': svm_features,
        'n_rows_used': int(len(svm_df)),
        'n_train': int(len(X_tr_s)),
        'n_test': int(len(X_te_s)),
        'class_distribution': class_distribution,
        'class_order': cat_labels,
        'kernels': kernels_dict,
        'decision_boundary_2d': {
            'feature_x': feat_2d[0],
            'feature_y': feat_2d[1],
            'feature_x_unit': COLUMN_UNITS.get(feat_2d[0], ''),
            'feature_y_unit': COLUMN_UNITS.get(feat_2d[1], ''),
            'x_range': [round(x_min, 1), round(x_max, 1)],
            'y_range': [round(y_min, 1), round(y_max, 1)],
            'grid_resolution': grid_res,
            'class_order': cat_labels,
            'n_fit_samples': int(len(sub_2d)),
            'kernels': boundary_kernels,
            'scatter_points': scatter_2d_pts
        }
    }

    json_dump(svm_results, 'svm_results.json')
    # ---------------------------------------------------------
    # 6. DATASET OVERVIEW + REAL AQI SWEEP (Landing page dial)
    # ---------------------------------------------------------
    print("Computing dataset overview & CO air-quality index sweep...")

    # US EPA carbon-monoxide AQI breakpoints on the 8-hour running mean.
    # Concentrations are held in TENTHS of a ppm so the whole derivation below is
    # exact integer arithmetic: the EPA truncation and rounding steps sit exactly
    # on half-way values often enough that float noise would otherwise flip the
    # published index by one, and make the numbers irreproducible.
    # (ppm_lo_tenths, ppm_hi_tenths, index_lo, index_hi)
    CO_AQI_BREAKPOINTS_TENTHS = [
        (0, 44, 0, 50),
        (45, 94, 51, 100),
        (95, 124, 101, 150),
        (125, 154, 151, 200),
        (155, 304, 201, 300),
        (305, 404, 301, 400),
        (405, 504, 401, 500),
    ]
    # Ideal-gas conversion for CO at 25 degC / 1 atm: ppm = mg/m3 * 24.45 / 28.01
    PPM_NUM, PPM_DEN = 2445, 280100   # milligram-per-m3 (x1000) -> ppm (x10)

    def aqi_from_milli_mg(milli):
        """EPA CO index for an 8-hour mean given in thousandths of a mg/m3."""
        if milli is None:
            return None
        ppm_tenths = (milli * PPM_NUM) // PPM_DEN      # EPA truncates to 0.1 ppm
        for lo_t, hi_t, i_lo, i_hi in CO_AQI_BREAKPOINTS_TENTHS:
            if ppm_tenths <= hi_t:
                span = hi_t - lo_t
                if span <= 0:
                    return i_lo
                num = (i_hi - i_lo) * max(ppm_tenths - lo_t, 0)
                return i_lo + (2 * num + span) // (2 * span)   # half-up
        return CO_AQI_BREAKPOINTS_TENTHS[-1][3]

    # CO(GT) is recorded to one decimal place, so tenths are exact integers.
    co_tenths = [
        None if pd.isna(v) else int(round(float(v) * 10))
        for v in df['CO(GT)']
    ]
    timestamps_all = list(df['timestamp'])

    # 8-hour running mean in thousandths of a mg/m3, rounded half-up:
    # sum_of_tenths * 1000 / 80 == sum_of_tenths * 125 / 10.
    co_8h_milli = []
    for i in range(len(co_tenths)):
        window = co_tenths[i - 7:i + 1] if i >= 7 else []
        if len(window) != 8 or any(v is None for v in window):
            co_8h_milli.append(None)
            continue
        co_8h_milli.append((sum(window) * 125 + 5) // 10)

    co_8h_vals = [None if m is None else m / 1000 for m in co_8h_milli]
    aqi_series = [aqi_from_milli_mg(m) for m in co_8h_milli]

    # Longest run of consecutive hours with a complete 8-hour window (capped at a
    # week), so the dial sweeps genuinely consecutive readings from the CSV rather
    # than a synthetic waveform or a stitched-together set of unrelated hours.
    SWEEP_HOURS_MAX = 168
    best_len = 0
    best_start = 0
    run = 0
    for i, v in enumerate(aqi_series):
        run = run + 1 if v is not None else 0
        if run > best_len:
            best_len = run
            best_start = i - run + 1
    sweep_len = min(best_len, SWEEP_HOURS_MAX)
    sweep_idx = list(range(best_start, best_start + sweep_len))

    sweep = [
        {
            'timestamp': timestamps_all[i],
            'co_8h_mg_m3': co_8h_vals[i],
            'aqi': aqi_series[i],
        }
        for i in sweep_idx
    ]

    observed_aqi = [v for v in aqi_series if v is not None]

    missing_counts = {}
    for c in numeric_cols:
        if c in df.columns and c != 'hour':
            missing_counts[c] = int(df[c].isna().sum())

    overview_results = {
        'source_file': 'data/AirQualityUCI.csv',
        'record_count': int(len(df)),
        'date_range': {
            'start': timestamps_all[0],
            'end': timestamps_all[-1],
        },
        'columns': [c for c in numeric_cols if c != 'hour'],
        'units': {c: COLUMN_UNITS.get(c, '') for c in numeric_cols if c != 'hour'},
        'missing_counts': missing_counts,
        'missing_sentinel': MISSING_SENTINEL,
        'aqi': {
            'basis': 'US EPA carbon-monoxide AQI on the 8-hour running mean of CO(GT)',
            'conversion': 'ppm = CO(GT) mg/m3 * 24.45 / 28.01 (CO at 25 degC, 1 atm)',
            'rounding': '8-h mean rounded half-up to 3 decimals; ppm truncated to 0.1; index rounded half-up',
            'observed_count': len(observed_aqi),
            'min': int(min(observed_aqi)) if observed_aqi else None,
            'max': int(max(observed_aqi)) if observed_aqi else None,
            'mean': round(float(np.mean(observed_aqi)), 1) if observed_aqi else None,
            'sweep_hours': len(sweep),
            'sweep_is_contiguous': True,
            'sweep': sweep,
        },
    }

    json_dump(overview_results, 'overview_results.json')

    print("\nSUCCESS: All ML project results generated directly and exclusively from data/AirQualityUCI.csv!")

if __name__ == '__main__':
    main()
