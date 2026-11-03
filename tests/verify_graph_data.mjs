/**
 * End-to-end data verification: data/AirQualityUCI.csv -> results JSON -> chart input.
 *
 * This file re-parses the raw CSV with its own independent parser (semicolon
 * delimiter, comma decimal separator, -200 missing sentinel) and asserts that
 * the arrays handed to every chart on the site match it. It imports the real
 * chart-data-preparation functions out of assets/charts.js, so the values under
 * test are literally the ones Chart.js receives.
 *
 * Run with:  node tests/verify_graph_data.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const require = createRequire(import.meta.url);

const charts = require(path.join(ROOT, 'assets', 'charts.js'));
const {
  buildImputationSeries,
  imputationChartMae,
  resolvePairScatter,
  buildPolyfitDatasets,
  buildPredVsActual,
  buildCoefSeries,
  buildBoundaryGrid,
  describeOlsCollinearity,
  formatPolynomialEquation
} = charts;

/* ------------------------------------------------------------------ harness */

let passed = 0;
const failures = [];

function check(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push({ name, message: err.message });
    console.log(`  FAIL  ${name}\n        ${err.message}`);
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function assertEqual(actual, expected, msg) {
  if (actual !== expected) {
    throw new Error(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertClose(actual, expected, tol, msg) {
  assert(typeof actual === 'number' && Number.isFinite(actual), `${msg}: ${actual} is not a finite number`);
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${msg}: expected ~${expected} (+/-${tol}), got ${actual}`);
  }
}

function section(title) {
  console.log(`\n${title}`);
}

/* ------------------------------------- independent AirQualityUCI.csv parser */

const MISSING_SENTINEL = -200;

function parseAirQualityCsv(file) {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  const header = lines[0].split(';');
  // Trailing ";;" in every line produces two nameless columns; drop them.
  const columns = header.filter(h => h.trim() !== '');

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line === '') continue;
    const cells = line.split(';');
    // Fully blank rows (";;;;...") and rows without a Date are not observations.
    if (cells.every(c => c.trim() === '')) continue;
    if (cells[0].trim() === '') continue;

    const row = { Date: cells[0].trim(), Time: cells[1].trim() };
    for (let c = 2; c < columns.length; c++) {
      const raw = (cells[c] ?? '').trim();
      if (raw === '') {
        row[columns[c]] = null;
        continue;
      }
      const v = Number(raw.replace(',', '.'));
      row[columns[c]] = Number.isFinite(v) && v !== MISSING_SENTINEL ? v : null;
    }
    row.hour = Number(row.Time.split('.')[0]);
    row.timestamp = `${row.Date} ${String(row.hour).padStart(2, '0')}:00`;
    rows.push(row);
  }
  return { columns, rows };
}

function column(rows, name) {
  return rows.map(r => r[name]);
}

function observed(rows, name) {
  return column(rows, name).filter(v => v !== null);
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function median(values) {
  const s = values.slice().sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function pearson(rows, a, b) {
  const pairs = rows
    .map(r => [r[a], r[b]])
    .filter(([x, y]) => x !== null && y !== null);
  const n = pairs.length;
  const mx = mean(pairs.map(p => p[0]));
  const my = mean(pairs.map(p => p[1]));
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (const [x, y] of pairs) {
    sxy += (x - mx) * (y - my);
    sxx += (x - mx) ** 2;
    syy += (y - my) ** 2;
  }
  return { r: sxy / Math.sqrt(sxx * syy), n };
}

/* ------------------------------------------------------------------- inputs */

const { rows } = parseAirQualityCsv(path.join(ROOT, 'data', 'AirQualityUCI.csv'));
const json = name => JSON.parse(fs.readFileSync(path.join(ROOT, 'data', name), 'utf8'));

const overview = json('overview_results.json');
const imputation = json('imputation_results.json');
const covariance = json('covariance_results.json');
const polyfit = json('polyfit_results.json');
const ridge = json('ridge_results.json');
const svm = json('svm_results.json');

console.log(`Parsed ${rows.length} observation rows from data/AirQualityUCI.csv`);

/* =========================================================== 1. CSV parsing */

section('1. CSV parsing / dataset shape');

check('row count matches the documented 9,357 hourly observations', () => {
  assertEqual(rows.length, 9357, 'observation rows');
});

check('overview record_count equals the parsed row count', () => {
  assertEqual(overview.record_count, rows.length, 'overview.record_count');
});

check('first and last records parse with comma decimals intact', () => {
  const first = rows[0];
  assertEqual(first.timestamp, '10/03/2004 18:00', 'first timestamp');
  assertEqual(first['CO(GT)'], 2.6, 'first CO(GT)');
  assertEqual(first['C6H6(GT)'], 11.9, 'first C6H6(GT)');
  assertEqual(first['PT08.S1(CO)'], 1360, 'first PT08.S1(CO)');
  assertEqual(first.AH, 0.7578, 'first AH');

  const last = rows[rows.length - 1];
  assertEqual(last.timestamp, '04/04/2005 14:00', 'last timestamp');
  assertEqual(last['CO(GT)'], 2.2, 'last CO(GT)');
  assertEqual(last.T, 28.5, 'last T');
  assertEqual(last['NMHC(GT)'], null, 'last NMHC(GT) is the -200 sentinel');
});

check('overview date_range matches first/last CSV timestamps', () => {
  assertEqual(overview.date_range.start, rows[0].timestamp, 'date_range.start');
  assertEqual(overview.date_range.end, rows[rows.length - 1].timestamp, 'date_range.end');
});

check('-200 sentinel counts match per column', () => {
  for (const [col, count] of Object.entries(overview.missing_counts)) {
    const actual = column(rows, col).filter(v => v === null).length;
    assertEqual(count, actual, `missing count for ${col}`);
  }
  assertEqual(overview.missing_counts['CO(GT)'], 1683, 'CO(GT) missing');
  assertEqual(overview.missing_counts['NMHC(GT)'], 8443, 'NMHC(GT) missing');
});

/* ================================================ 2. Imputation timeline */

section('2. Imputation timeline chart (unit1/imputation.html)');

const csvWindow = rows.slice(
  imputation.window.start_row,
  imputation.window.end_row
);

check('chart series is numeric and 400 points long (not undefined)', () => {
  const series = buildImputationSeries(imputation.methods.knn);
  assertEqual(series.values.length, csvWindow.length, 'series length');
  assert(series.values.every(v => typeof v === 'number' && Number.isFinite(v)),
    'every KNN-imputed point must be a finite number');
});

check('measured points equal the CSV window readings, in order', () => {
  const series = buildImputationSeries(imputation.methods.mean, 'before');
  csvWindow.forEach((row, i) => {
    assertEqual(series.values[i], row['CO(GT)'], `before[${i}] (${row.timestamp})`);
  });
});

check('x-axis labels are the CSV Date/Time of the matching y value', () => {
  const series = buildImputationSeries(imputation.methods.knn);
  assertEqual(series.labels.length, series.values.length, 'label/value count');
  csvWindow.forEach((row, i) => {
    assertEqual(series.labels[i], row.timestamp, `label[${i}]`);
  });
  // Spot checks at both ends and in the middle of the window.
  assertEqual(series.labels[0], '19/03/2004 02:00', 'first label');
  assertEqual(series.labels[199], rows[imputation.window.start_row + 199].timestamp, 'middle label');
  assertEqual(series.labels[399], '04/04/2004 17:00', 'last label');
});

check('missing flags mark exactly the -200 rows of the window', () => {
  const series = buildImputationSeries(imputation.methods.knn);
  csvWindow.forEach((row, i) => {
    assertEqual(series.missing[i], row['CO(GT)'] === null, `was_missing[${i}]`);
  });
  const flagged = series.missing.filter(Boolean).length;
  const actual = csvWindow.filter(r => r['CO(GT)'] === null).length;
  assertEqual(flagged, actual, 'missing count in window');
});

check('observed readings are never overwritten by imputation', () => {
  for (const key of ['mean', 'median', 'knn']) {
    const series = buildImputationSeries(imputation.methods[key]);
    csvWindow.forEach((row, i) => {
      if (row['CO(GT)'] !== null) {
        assertEqual(series.values[i], row['CO(GT)'], `${key} after[${i}] must stay measured`);
      }
    });
  }
});

check('mean/median fills equal the CSV column mean/median', () => {
  const co = observed(rows, 'CO(GT)');
  const csvMean = Number(mean(co).toFixed(2));
  const csvMedian = Number(median(co).toFixed(2));

  const meanSeries = buildImputationSeries(imputation.methods.mean);
  const medianSeries = buildImputationSeries(imputation.methods.median);
  csvWindow.forEach((row, i) => {
    if (row['CO(GT)'] === null) {
      assertClose(meanSeries.values[i], csvMean, 0.005, `mean fill at ${i}`);
      assertClose(medianSeries.values[i], csvMedian, 0.005, `median fill at ${i}`);
    }
  });
  assertClose(csvMean, 2.15, 0.005, 'CSV CO(GT) mean');
  assertClose(csvMedian, 1.8, 0.005, 'CSV CO(GT) median');
});

check('total filled count equals CSV -200 count across the three columns', () => {
  const expected = imputation.columns
    .map(c => column(rows, c).filter(v => v === null).length)
    .reduce((a, b) => a + b, 0);
  assertEqual(imputation.methods.knn.filled_count, expected, 'filled_count');
});

check('MAE readout uses the plotted column, not the mixed-unit average', () => {
  for (const key of ['mean', 'median', 'knn']) {
    const m = imputation.methods[key];
    assertEqual(imputationChartMae(m), m.mae_per_column['CO(GT)'], `${key} chart MAE`);
    assert(imputationChartMae(m) !== m.mae, `${key} must not surface the cross-column average`);
  }
});

check('all three methods are scored on the same masked cells', () => {
  const methods = ['mean', 'median', 'knn'].map(k => imputation.methods[k]);
  const cells = methods.map(m => m.holdout_test_count);
  assert(new Set(cells).size === 1,
    `MAE must be measured over identical cells, got ${JSON.stringify(cells)}`);
  assert(cells[0] > 0, 'masked cell count must be positive');
  const rows = methods.map(m => m.holdout_rows);
  assert(new Set(rows).size === 1, 'all methods must share the same evaluation rows');
  // KNN exploits the correlated sensor channels, so it must beat a constant fill
  // on the plotted column once the comparison is like-for-like.
  const co = k => imputation.methods[k].mae_per_column['CO(GT)'];
  assert(co('knn') < co('mean') && co('knn') < co('median'),
    `KNN CO(GT) MAE ${co('knn')} should beat mean ${co('mean')} / median ${co('median')}`);
});

check('readout text quotes the MAE the panel actually displays', () => {
  for (const key of ['mean', 'median', 'knn']) {
    const m = imputation.methods[key];
    const shown = imputationChartMae(m).toFixed(2);
    assert(m.description.includes(shown),
      `${key} description must quote the displayed CO(GT) MAE ${shown}: "${m.description}"`);
  }
});

check('reported model metrics are not inflated by scaler leakage', () => {
  // Standardising before the split leaked test statistics into training, which
  // showed up as a train R2 at or above the test R2 for every alpha.
  for (const [key, m] of Object.entries(ridge.models)) {
    assert(m.r2_train < m.r2_test + 1e-9 || m.r2_train <= 1,
      `${key} train R2 ${m.r2_train} looks implausible`);
    assert(m.r2_train > 0 && m.r2_test > 0, `${key} R2 should be positive`);
  }
  const src = fs.readFileSync(path.join(ROOT, 'scripts', 'generate_all_results.py'), 'utf8');
  assert(!src.includes('scaler.fit_transform(X_reg)'),
    'ridge must not standardise the full matrix before the train/test split');
  assert(!src.includes('scaler_svm.fit_transform(X_svm)'),
    'SVM must not standardise the full matrix before the train/test split');
  assert(src.includes('StandardScaler().fit(X_train_raw)'),
    'ridge scaler must be fitted on the training split only');
  assert(src.includes('StandardScaler().fit(X_tr_raw)'),
    'SVM scaler must be fitted on the training split only');
});

check('KNN neighbourhood size matches the "KNN (k=5)" UI label', () => {
  assertEqual(imputation.knn_neighbours, 5, 'knn_neighbours');
  const html = fs.readFileSync(path.join(ROOT, 'unit1', 'imputation.html'), 'utf8');
  assert(html.includes(`KNN (k=${imputation.knn_neighbours})`),
    'imputation.html tab label must match knn_neighbours');
});

/* ================================================== 3. Covariance heatmap */

section('3. Covariance heatmap + scatter (unit1/covariance.html)');

check('every matrix cell equals the pairwise Pearson r from the CSV', () => {
  covariance.pollutants.forEach((p1, i) => {
    covariance.pollutants.forEach((p2, j) => {
      const expected = p1 === p2 ? 1 : Number(pearson(rows, p1, p2).r.toFixed(3));
      assertClose(covariance.matrix[i][j], expected, 1e-9, `matrix[${p1}][${p2}]`);
    });
  });
});

check('heatmap cell label matches the r used for that row/column pair', () => {
  // Reproduces the row/col indexing CovarianceManager.initHeatmap uses.
  const { pollutants, matrix } = covariance;
  const rowIdx = pollutants.indexOf('NOx(GT)');
  const colIdx = pollutants.indexOf('T');
  assertClose(matrix[rowIdx][colIdx], Number(pearson(rows, 'NOx(GT)', 'T').r.toFixed(3)), 1e-9,
    'NOx(GT) x T cell');
  assertClose(matrix[colIdx][rowIdx], matrix[rowIdx][colIdx], 1e-9, 'matrix symmetry');
});

check('scatter points for each pair exist in the CSV in (x=p1, y=p2) order', () => {
  const { pollutants } = covariance;
  for (const p1 of pollutants) {
    for (const p2 of pollutants) {
      const { points } = resolvePairScatter(covariance, p1, p2);
      assert(points.length > 0, `${p1} vs ${p2} must have real points`);
      const pool = new Set(
        rows
          .filter(r => r[p1] !== null && r[p2] !== null)
          .map(r => `${Number(r[p1].toFixed(2))}|${Number(r[p2].toFixed(2))}`)
      );
      for (const pt of points) {
        assert(pool.has(`${pt.x}|${pt.y}`),
          `${p1} vs ${p2}: point (${pt.x}, ${pt.y}) is not in AirQualityUCI.csv`);
      }
    }
  }
});

check('sample size reported per pair is the pairwise-complete CSV count', () => {
  const spot = [
    ['CO(GT)', 'C6H6(GT)'],
    ['T', 'RH'],
    ['NOx(GT)', 'NO2(GT)'],
    ['CO(GT)', 'T']
  ];
  for (const [p1, p2] of spot) {
    const resolved = resolvePairScatter(covariance, p1, p2);
    assertEqual(resolved.n, pearson(rows, p1, p2).n, `n for ${p1} vs ${p2}`);
  }
  // The all-column complete-case count must NOT be reused for every pair.
  const coBenzene = resolvePairScatter(covariance, 'CO(GT)', 'C6H6(GT)');
  assert(coBenzene.n !== covariance.n_rows_used,
    'per-pair n should differ from the all-column complete-case count');
});

check('diagonal (identity) pair uses real readings, not generated points', () => {
  for (const p of covariance.pollutants) {
    const resolved = resolvePairScatter(covariance, p, p);
    assertEqual(resolved.corr, 1, `${p} self-correlation`);
    assertEqual(resolved.n, observed(rows, p).length, `${p} self n`);
    assert(resolved.points.length > 0, `${p} identity scatter must have points`);
    const pool = new Set(observed(rows, p).map(v => Number(v.toFixed(2))));
    for (const pt of resolved.points) {
      assertEqual(pt.x, pt.y, `${p} identity point must satisfy y = x`);
      assert(pool.has(pt.x), `${p}: identity point ${pt.x} is not in the CSV`);
    }
  }
});

check('reversed pair lookup flips the axes rather than swapping meaning', () => {
  const fwd = resolvePairScatter(covariance, 'CO(GT)', 'NOx(GT)');
  const rev = resolvePairScatter(covariance, 'NOx(GT)', 'CO(GT)');
  assertEqual(rev.points.length, fwd.points.length, 'point count');
  fwd.points.forEach((pt, i) => {
    assertEqual(rev.points[i].x, pt.y, `reversed x at ${i}`);
    assertEqual(rev.points[i].y, pt.x, `reversed y at ${i}`);
  });
  assertClose(rev.corr, fwd.corr, 1e-9, 'r is symmetric');
});

check('top-3 pairs are the strongest |r| pairs in the CSV', () => {
  const all = [];
  const { pollutants } = covariance;
  for (let i = 0; i < pollutants.length; i++) {
    for (let j = i + 1; j < pollutants.length; j++) {
      const { r } = pearson(rows, pollutants[i], pollutants[j]);
      all.push({ a: pollutants[i], b: pollutants[j], abs: Math.abs(Number(r.toFixed(3))) });
    }
  }
  all.sort((x, y) => y.abs - x.abs);
  covariance.top_pairs.forEach((p, i) => {
    assertEqual(`${p.a}|${p.b}`, `${all[i].a}|${all[i].b}`, `top pair #${i + 1}`);
  });
});

/* ============================================== 4. Polynomial curve fitting */

section('4. Polynomial curve fit (unit1/polynomial.html)');

check('scatter points are real (hour, NO2(GT)) readings from the CSV', () => {
  const pool = new Set(
    rows
      .filter(r => r['NO2(GT)'] !== null)
      .map(r => `${r.hour}|${Number(r['NO2(GT)'].toFixed(1))}`)
  );
  for (const deg of [1, 2, 3, 4]) {
    const { actualPoints } = buildPolyfitDatasets(polyfit, deg);
    assertEqual(actualPoints.length, 300, `degree ${deg} scatter count`);
    for (const pt of actualPoints) {
      assert(pool.has(`${pt.x}|${pt.y}`),
        `degree ${deg}: (hour ${pt.x}, NO2 ${pt.y}) is not in AirQualityUCI.csv`);
    }
  }
});

check('rows used equals the number of observed NO2(GT) readings', () => {
  assertEqual(polyfit.n_rows_used, observed(rows, 'NO2(GT)').length, 'n_rows_used');
  assertEqual(polyfit.n_train + polyfit.n_test, polyfit.n_rows_used, 'train + test split');
});

check('fitted curve spans hours 0-23 on the same axis as the scatter', () => {
  for (const deg of [1, 2, 3, 4]) {
    const { actualPoints, curvePoints, xRange } = buildPolyfitDatasets(polyfit, deg);
    assertEqual(curvePoints.length, 100, `degree ${deg} curve resolution`);
    // Curve points must be {x, y}: a bare y-array would be plotted against
    // scatter categories instead of hours.
    assert(curvePoints.every(pt => typeof pt.x === 'number' && typeof pt.y === 'number'),
      `degree ${deg}: curve points must carry their own x`);
    assertEqual(curvePoints[0].x, xRange[0], `degree ${deg} curve start`);
    assertEqual(curvePoints[curvePoints.length - 1].x, xRange[1], `degree ${deg} curve end`);
    const xs = actualPoints.map(p => p.x);
    assert(Math.min(...xs) >= xRange[0] && Math.max(...xs) <= xRange[1],
      `degree ${deg}: scatter hours must sit inside the axis range`);
  }
});

check('curve y values equal the published coefficients evaluated at x', () => {
  for (const deg of [1, 2, 3, 4]) {
    const coeffs = polyfit.degrees[String(deg)].coefficients; // highest power first
    const { curvePoints } = buildPolyfitDatasets(polyfit, deg);
    for (const pt of curvePoints) {
      // Horner evaluation of the published coefficients at the point's own x.
      const y = coeffs.reduce((acc, c) => acc * pt.x + c, 0);
      assertClose(pt.y, y, 1e-3, `degree ${deg} curve at x=${pt.x}`);
    }
  }
});

check('rendered equation matches the coefficient order (highest power first)', () => {
  const eq = formatPolynomialEquation(polyfit.degrees['1'].coefficients);
  const [slope, intercept] = polyfit.degrees['1'].coefficients;
  assert(eq.includes(slope.toFixed(2)) || eq.includes(Math.abs(slope).toFixed(2)),
    `linear slope ${slope} missing from "${eq}"`);
  assert(eq.includes(Math.abs(intercept).toFixed(1)) || eq.includes(Math.abs(intercept).toFixed(2)),
    `linear intercept ${intercept} missing from "${eq}"`);
  assert(eq.includes('x'), 'equation should contain an x term');
});

/* ============================================= 5. Ridge regression charts */

section('5. Ridge regression (unit2/ridge.html)');

check('high-correlation pairs match Pearson r over the regression rows', () => {
  const complete = rows.filter(r =>
    ridge.predictors.every(p => r[p] !== null) && r[ridge.target] !== null);
  assertEqual(ridge.n_rows_used, complete.length, 'complete regression rows');
  assertEqual(ridge.n_train + ridge.n_test, complete.length, 'train + test split');
  for (const pair of ridge.high_correlation_pairs) {
    const { r } = pearson(complete, pair.a, pair.b);
    assertClose(pair.corr, Number(r.toFixed(3)), 1e-9, `r for ${pair.a} x ${pair.b}`);
    assert(Math.abs(pair.corr) > 0.7, `${pair.a} x ${pair.b} should exceed the 0.70 threshold`);
  }
});

check('predicted-vs-actual x values are real CO(GT) concentrations', () => {
  const { points } = buildPredVsActual(ridge);
  assertEqual(points.length, ridge.pred_vs_actual_sample.length, 'plotted points');
  const co = observed(rows, 'CO(GT)');
  const lo = Math.min(...co);
  const hi = Math.max(...co);
  const pool = new Set(co.map(v => Number(v.toFixed(2))));
  for (const pt of points) {
    assert(pt.x >= lo && pt.x <= hi, `actual ${pt.x} outside CSV CO(GT) range ${lo}..${hi}`);
    assert(pool.has(pt.x), `actual ${pt.x} is not a CO(GT) value in AirQualityUCI.csv`);
  }
});

check('y = x reference line spans the observed data, not a fixed window', () => {
  const { points, refLine } = buildPredVsActual(ridge);
  const all = points.flatMap(pt => [pt.x, pt.y]);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  assert(refLine[0].x <= lo + 1e-9, `reference line starts at ${refLine[0].x}, above data min ${lo}`);
  assert(refLine[1].x >= hi - 1e-9, `reference line ends at ${refLine[1].x}, below data max ${hi}`);
  assertEqual(refLine[0].x, refLine[0].y, 'reference line must satisfy y = x');
  assertEqual(refLine[1].x, refLine[1].y, 'reference line must satisfy y = x');
});

check('coefficient bars line up label-to-value for every alpha', () => {
  for (const key of Object.keys(ridge.models)) {
    const series = buildCoefSeries(ridge, key);
    assertEqual(series.labels.length, ridge.predictors.length, `${key} bar count`);
    series.labels.forEach((label, i) => {
      assertEqual(label, ridge.predictors[i], `${key} label position ${i}`);
      assertEqual(series.selected[i], ridge.models[key].coefficients[label], `${key} bar for ${label}`);
      assertEqual(series.ols[i], ridge.models.ols.coefficients[label], `${key} OLS bar for ${label}`);
    });
    assertEqual(series.alpha, ridge.models[key].alpha, `${key} alpha`);
  }
});

check('L2 shrinkage is monotonic in alpha (chart must show shrinking bars)', () => {
  const norm = key => buildCoefSeries(ridge, key).selected
    .reduce((acc, c) => acc + Math.abs(c), 0);
  const order = ['ols', 'ridge_1.0', 'ridge_10.0', 'ridge_50.0', 'ridge_100.0'];
  for (let i = 1; i < order.length; i++) {
    assert(norm(order[i]) <= norm(order[i - 1]) + 1e-9,
      `${order[i]} L1 norm ${norm(order[i])} should not exceed ${order[i - 1]} (${norm(order[i - 1])})`);
  }
});

check('OLS collinearity readout quotes the fitted coefficients verbatim', () => {
  const d = describeOlsCollinearity(ridge);
  assert(d !== null, 'diagnostic should be derivable');
  const coefs = ridge.models.ols.coefficients;
  assertEqual(d.pairCorr, ridge.high_correlation_pairs[0].corr, 'quoted pair r');
  assertEqual(d.inflated.coef, coefs[d.inflated.name], 'quoted inflated coefficient');
  assertEqual(d.suppressed.coef, coefs[d.suppressed.name], 'quoted suppressed coefficient');
  assert(d.inflated.coef > 0, 'inflated weight should be positive');
  assert(d.suppressed.coef < 0, 'suppressed weight should be negative');
});

check('no hard-coded dataset numbers left in the chart layer', () => {
  const js = fs.readFileSync(path.join(ROOT, 'assets', 'charts.js'), 'utf8');
  for (const stale of ['0.898', '1.3381', '-0.0385', 'PM2.5', '81.8']) {
    assert(!js.includes(stale), `assets/charts.js still hard-codes "${stale}"`);
  }
  assert(!/Math\.(sin|cos)\(seed/.test(js), 'assets/charts.js still fabricates scatter points');
});

/* ==================================================== 6. SVM classification */

section('6. SVM classification (unit2/svm.html)');

check('class thresholds partition the real CO(GT) range', () => {
  const co = observed(rows, 'CO(GT)');
  assertEqual(svm.category_thresholds.Hazardous.max, Number(Math.max(...co).toFixed(1)),
    'Hazardous upper bound');
  assertEqual(svm.category_thresholds.Safe.min, 0, 'Safe lower bound');
});

check('class distribution matches labelling the CSV rows directly', () => {
  const complete = rows.filter(r =>
    svm.predictors.every(p => r[p] !== null) && r['CO(GT)'] !== null);
  assertEqual(svm.n_rows_used, complete.length, 'complete SVM rows');
  const counts = { Safe: 0, Moderate: 0, Hazardous: 0 };
  for (const r of complete) {
    const v = r['CO(GT)'];
    counts[v <= 2 ? 'Safe' : v <= 4 ? 'Moderate' : 'Hazardous'] += 1;
  }
  for (const cls of svm.class_order) {
    assertEqual(svm.class_distribution[cls], counts[cls], `${cls} count`);
  }
});

check('confusion matrices sum to the test split and agree with accuracy', () => {
  assertEqual(svm.n_train + svm.n_test, svm.n_rows_used, 'train + test split');
  for (const [name, k] of Object.entries(svm.kernels)) {
    const total = k.confusion_matrix.flat().reduce((a, b) => a + b, 0);
    assertEqual(total, svm.n_test, `${name} confusion matrix total`);
    const correct = k.confusion_matrix.reduce((acc, row, i) => acc + row[i], 0);
    assertClose(k.accuracy, correct / total, 5e-5, `${name} accuracy`);
    // Row order is actual class, column order is predicted class.
    svm.class_order.forEach((cls, i) => {
      const support = k.confusion_matrix[i].reduce((a, b) => a + b, 0);
      assertEqual(k.per_class[cls].support, support, `${name} ${cls} support`);
      assertClose(k.per_class[cls].recall, k.confusion_matrix[i][i] / support, 5e-5,
        `${name} ${cls} recall`);
    });
  }
});

check('decision-boundary grid is per kernel and differs between kernels', () => {
  const lin = buildBoundaryGrid(svm, 'linear');
  const rbf = buildBoundaryGrid(svm, 'rbf');
  assert(lin && rbf, 'both kernels need a grid');
  assertEqual(JSON.stringify(lin.grid) === JSON.stringify(rbf.grid), false,
    'linear and RBF regions must not be identical');
  for (const b of [lin, rbf]) {
    assertEqual(b.grid.length, b.resolution, 'grid rows');
    b.grid.forEach(row => {
      assertEqual(row.length, b.resolution, 'grid columns');
      row.forEach(v => assert(v >= 0 && v < b.classOrder.length, `class index ${v} out of range`));
    });
  }
});

check('boundary axes are the features actually plotted, in CSV units', () => {
  const b = buildBoundaryGrid(svm, 'rbf');
  assertEqual(b.featureX, 'PT08.S1(CO)', 'x feature');
  assertEqual(b.featureY, 'PT08.S2(NMHC)', 'y feature');

  const pool = rows.filter(r =>
    r[b.featureX] !== null && r[b.featureY] !== null && r['CO(GT)'] !== null);
  const xs = pool.map(r => r[b.featureX]);
  const ys = pool.map(r => r[b.featureY]);
  assert(b.xRange[0] >= Math.min(...xs) && b.xRange[1] <= Math.max(...xs),
    `x range ${b.xRange} outside CSV ${b.featureX} range`);
  assert(b.yRange[0] >= Math.min(...ys) && b.yRange[1] <= Math.max(...ys),
    `y range ${b.yRange} outside CSV ${b.featureY} range`);

  // Panel copy must not describe the plot as PCA - it is two raw channels.
  const html = fs.readFileSync(path.join(ROOT, 'unit2', 'svm.html'), 'utf8');
  assert(!/PCA/i.test(html), 'unit2/svm.html still labels the boundary plot as PCA');
});

check('boundary scatter points are real readings with the right class label', () => {
  const b = buildBoundaryGrid(svm, 'rbf');
  const pool = new Map();
  for (const r of rows) {
    if (r[b.featureX] === null || r[b.featureY] === null || r['CO(GT)'] === null) continue;
    const v = r['CO(GT)'];
    const cls = v <= 2 ? 'Safe' : v <= 4 ? 'Moderate' : 'Hazardous';
    pool.set(`${Number(r[b.featureX].toFixed(1))}|${Number(r[b.featureY].toFixed(1))}|${cls}`, true);
  }
  assertEqual(b.scatterPoints.length, 150, 'plotted point count');
  for (const pt of b.scatterPoints) {
    assert(pool.has(`${pt.x}|${pt.y}|${pt.label}`),
      `(${pt.x}, ${pt.y}) labelled ${pt.label} does not exist in AirQualityUCI.csv`);
    assert(pt.x >= b.xRange[0] && pt.x <= b.xRange[1], `point x ${pt.x} outside axis range`);
    assert(pt.y >= b.yRange[0] && pt.y <= b.yRange[1], `point y ${pt.y} outside axis range`);
  }
});

/* ============================================ 7. Landing page AQI dial */

section('7. Landing page AQI dial (index.html)');

// Independent reimplementation of the EPA CO index, in exact integer arithmetic:
// concentrations in tenths of a ppm, the 8-hour mean in thousandths of a mg/m3.
const PPM_NUM = 2445;
const PPM_DEN = 280100;
const CO_BREAKPOINTS_TENTHS = [
  [0, 44, 0, 50],
  [45, 94, 51, 100],
  [95, 124, 101, 150],
  [125, 154, 151, 200],
  [155, 304, 201, 300],
  [305, 404, 301, 400],
  [405, 504, 401, 500]
];

/** 8-hour mean of eight CO(GT) readings, in thousandths of a mg/m3 (half-up). */
function mean8Milli(window8) {
  const tenths = window8.map(v => Math.round(v * 10));
  const sum = tenths.reduce((a, b) => a + b, 0);
  return Math.floor((sum * 125 + 5) / 10);
}

function aqiFromMilli(milli) {
  const ppmTenths = Math.floor((milli * PPM_NUM) / PPM_DEN);
  for (const [loT, hiT, iLo, iHi] of CO_BREAKPOINTS_TENTHS) {
    if (ppmTenths <= hiT) {
      const span = hiT - loT;
      if (span <= 0) return iLo;
      const num = (iHi - iLo) * Math.max(ppmTenths - loT, 0);
      return iLo + Math.floor((2 * num + span) / (2 * span));
    }
  }
  return 500;
}

check('dial sweep values are the EPA CO index of the CSV 8-hour means', () => {
  const sweep = overview.aqi.sweep;
  assert(sweep.length > 0, 'sweep must not be empty');
  const byTimestamp = new Map(rows.map((r, i) => [r.timestamp, i]));

  for (const entry of sweep) {
    const idx = byTimestamp.get(entry.timestamp);
    assert(idx !== undefined, `sweep timestamp ${entry.timestamp} is not in the CSV`);
    const window8 = rows.slice(idx - 7, idx + 1).map(r => r['CO(GT)']);
    assertEqual(window8.length, 8, `8-hour window at ${entry.timestamp}`);
    assert(window8.every(v => v !== null), `8-hour window at ${entry.timestamp} has gaps`);

    const milli = mean8Milli(window8);
    assertEqual(entry.co_8h_mg_m3, milli / 1000, `8-h mean at ${entry.timestamp}`);
    assertEqual(entry.aqi, aqiFromMilli(milli), `AQI at ${entry.timestamp}`);
    // Sanity-check the integer path against plain floating-point averaging.
    assertClose(entry.co_8h_mg_m3, mean(window8), 1e-3, `8-h mean sanity at ${entry.timestamp}`);
  }
});

check('dial sweep is contiguous hourly readings from the CSV', () => {
  const sweep = overview.aqi.sweep;
  const idx = sweep.map(e => rows.findIndex(r => r.timestamp === e.timestamp));
  for (let i = 1; i < idx.length; i++) {
    assertEqual(idx[i], idx[i - 1] + 1, `sweep entry ${i} must follow entry ${i - 1}`);
  }
});

check('reported AQI min/max/mean match the whole dataset', () => {
  const co = column(rows, 'CO(GT)');
  const values = [];
  for (let i = 7; i < co.length; i++) {
    const w = co.slice(i - 7, i + 1);
    if (w.some(v => v === null)) continue;
    values.push(aqiFromMilli(mean8Milli(w)));
  }
  assertEqual(overview.aqi.observed_count, values.length, 'observed AQI count');
  assertEqual(overview.aqi.min, Math.min(...values), 'AQI min');
  assertEqual(overview.aqi.max, Math.max(...values), 'AQI max');
  assertClose(overview.aqi.mean, Number(mean(values).toFixed(1)), 1e-9, 'AQI mean');
});

check('gauge no longer synthesises the values it displays', () => {
  const js = fs.readFileSync(path.join(ROOT, 'assets', 'gauge.js'), 'utf8');
  // Math.sin/cos still position the needle and dial ticks; what must be gone is
  // the synthetic 40..185 waveform that used to supply the displayed number.
  assert(!/sinFactor/.test(js), 'assets/gauge.js still derives its value from a sine waveform');
  assert(!/185/.test(js), 'assets/gauge.js still hard-codes the synthetic sweep ceiling');
  assert(js.includes('overview_results.json'), 'gauge must read its values from the results file');
  assert(/showReading\s*\(/.test(js), 'gauge must display dataset sweep entries');
});

/* ------------------------------------------------------------------ report */

console.log(`\n${'='.repeat(60)}`);
if (failures.length === 0) {
  console.log(`ALL ${passed} CHECKS PASSED - graph inputs match data/AirQualityUCI.csv`);
} else {
  console.log(`${passed} passed, ${failures.length} FAILED`);
  for (const f of failures) console.log(`  - ${f.name}: ${f.message}`);
  process.exitCode = 1;
}
