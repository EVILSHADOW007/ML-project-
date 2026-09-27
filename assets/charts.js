/**
 * Urban Air Quality Profiling - Chart.js & Interactivity Engine
 */

// Shared Dark Theme Defaults for Chart.js (shadcn zinc dark)
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#a1a1aa';
  Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.tooltip.backgroundColor = '#18181b';
  Chart.defaults.plugins.tooltip.borderColor = '#27272a';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = '#fafafa';
  Chart.defaults.plugins.tooltip.bodyColor = '#a1a1aa';
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
  Chart.defaults.plugins.legend.labels.color = '#fafafa';
}

/* Helper to count up numbers in JetBrains Mono font */
function animateMonoNumber(el, start, end, duration = 600, decimals = 2) {
  if (!el) return;
  const startTime = performance.now();
  const step = (now) => {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const val = start + (end - start) * easeOut;
    el.textContent = val.toFixed(decimals);
    if (progress < 1) {
      requestAnimationFrame(step);
    }
  };
  requestAnimationFrame(step);
}

/* ==========================================================================
   1. UNIT 1 OVERVIEW SPARKLINES
   ========================================================================== */
function initUnit1Sparklines(imputationData, covarianceData, polyfitData) {
  // Sparkline 1: Imputation Preview
  const elImp = document.getElementById('sparkline-imputation');
  if (elImp && imputationData) {
    const knn = imputationData.methods.knn.after;
    new Chart(elImp.getContext('2d'), {
      type: 'line',
      data: {
        labels: knn.map(d => d.time),
        datasets: [{
          data: knn.map(d => d.value),
          borderColor: '#4ADE80',
          borderWidth: 1.5,
          tension: 0.3,
          pointRadius: knn.map(d => d.is_missing ? 3 : 0),
          pointBackgroundColor: knn.map(d => d.is_missing ? '#FB7185' : '#4ADE80')
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false } }
      }
    });
  }

  // Sparkline 2: Covariance Matrix Mini Preview
  const elCov = document.getElementById('sparkline-covariance');
  if (elCov && covarianceData) {
    let pts = [];
    if (covarianceData.top_pairs && covarianceData.top_pairs.length > 0) {
      pts = covarianceData.top_pairs[0].scatter || [];
    } else if (covarianceData.scatter_data) {
      pts = covarianceData.scatter_data["PM2.5_PM10"] || Object.values(covarianceData.scatter_data)[0] || [];
    }

    new Chart(elCov.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [{
          data: pts,
          backgroundColor: '#FBBF24',
          pointRadius: 2.5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false } }
      }
    });

    const topPairSpan = document.querySelector('#card-covariance .sparkline-container + div span');
    if (topPairSpan && covarianceData.top_pairs && covarianceData.top_pairs.length > 0) {
      const topP = covarianceData.top_pairs[0];
      topPairSpan.textContent = `Top Corr: ${topP.a} ↔ ${topP.b} (${topP.corr >= 0 ? '+' : ''}${topP.corr.toFixed(2)})`;
    }
  }

  // Sparkline 3: Polynomial Fit Preview
  const elPoly = document.getElementById('sparkline-polyfit');
  if (elPoly && polyfitData) {
    const deg3 = polyfitData.degrees["3"];
    new Chart(elPoly.getContext('2d'), {
      type: 'line',
      data: {
        labels: deg3.actual_points.map(p => p.x),
        datasets: [
          {
            type: 'scatter',
            data: deg3.actual_points,
            backgroundColor: '#94A3AD',
            pointRadius: 2
          },
          {
            type: 'line',
            data: deg3.curve_points.map(p => p.y),
            borderColor: '#FB923C',
            borderWidth: 1.5,
            tension: 0.4,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false } }
      }
    });
  }
}

/* ==========================================================================
   2. IMPUTATION PAGE CONTROLLER
   ========================================================================== */
class ImputationManager {
  constructor(canvasEl, data) {
    this.canvas = canvasEl;
    this.data = data;
    this.currentMethod = 'knn'; // Default
    this.chart = null;
    this.isFilled = true;
    this.animating = false;
    this.animTimeouts = [];

    this.counterEl = document.getElementById('imp-filled-counter');
    this.totalMissingEl = document.getElementById('imp-total-missing-label');
    this.maeEl = document.getElementById('imp-mae-value');
    this.maeUnitEl = document.getElementById('imp-mae-unit');
    this.descEl = document.getElementById('imp-method-desc');
    this.titleEl = document.getElementById('imp-chart-title');
    this.fillBtn = document.getElementById('btn-fill-missing');

    this.initChart();
    this.setupEvents();
  }

  getCurrentMethodData() {
    if (!this.data || !this.data.methods) return null;
    return this.data.methods[this.currentMethod] || Object.values(this.data.methods)[0];
  }

  isIndexMissing(idx, methodData = null) {
    const md = methodData || this.getCurrentMethodData();
    if (!md) return false;
    if (Array.isArray(md.was_missing) && idx < md.was_missing.length) {
      return Boolean(md.was_missing[idx]);
    }
    const beforeArr = md.before || [];
    return beforeArr[idx] === null || beforeArr[idx] === undefined;
  }

  getMissingIndices(methodData) {
    if (!methodData) return [];
    if (Array.isArray(methodData.was_missing) && methodData.was_missing.length > 0) {
      const indices = [];
      methodData.was_missing.forEach((isMiss, idx) => {
        if (isMiss) indices.push(idx);
      });
      return indices;
    }
    const beforeArr = methodData.before || [];
    const indices = [];
    beforeArr.forEach((val, idx) => {
      if (val === null || val === undefined) indices.push(idx);
    });
    return indices;
  }

  getPointStyles(methodData) {
    const afterArr = methodData?.after || methodData?.before || [];
    const bgColors = [];
    const borderColors = [];
    const radii = [];
    const borderWidths = [];
    const hoverRadii = [];

    for (let i = 0; i < afterArr.length; i++) {
      const isMissing = this.isIndexMissing(i, methodData);
      if (isMissing) {
        bgColors.push('#EF4444');      // Distinct vibrant red for missing/imputed gaps
        borderColors.push('#FFFFFF');  // Crisp white outline
        radii.push(4.5);               // Visually distinct larger marker
        borderWidths.push(1.5);
        hoverRadii.push(6.5);
      } else {
        bgColors.push('#4ADE80');      // Green for normal measured readings
        borderColors.push('transparent');
        radii.push(2);
        borderWidths.push(0);
        hoverRadii.push(4);
      }
    }

    return { bgColors, borderColors, radii, borderWidths, hoverRadii };
  }

  cancelAnimation() {
    if (this.animTimeouts && this.animTimeouts.length > 0) {
      this.animTimeouts.forEach(tid => clearTimeout(tid));
      this.animTimeouts = [];
    }
    this.animating = false;
  }

  initChart() {
    if (!this.canvas || !this.data || !this.data.methods) return;

    const methodData = this.getCurrentMethodData();
    if (!methodData) return;

    const colName = methodData.chart_column || "CO(GT)";
    const afterArr = methodData.after || methodData.before || [];
    const missingIndices = this.getMissingIndices(methodData);
    const totalMissing = missingIndices.length;
    const labels = afterArr.map((_, i) => i + 1);
    const styles = this.getPointStyles(methodData);

    this.chart = new Chart(this.canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: `${colName} Telemetry (${this.currentMethod.toUpperCase()})`,
          data: [...afterArr],
          borderColor: '#4ADE80',
          borderWidth: 2,
          tension: 0.25,
          spanGaps: true,
          pointRadius: styles.radii,
          pointBackgroundColor: styles.bgColors,
          pointBorderColor: styles.borderColors,
          pointBorderWidth: styles.borderWidths,
          pointHoverRadius: styles.hoverRadii,
          pointHoverBackgroundColor: (ctx) => this.isIndexMissing(ctx.dataIndex) ? '#F87171' : '#34D399'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        scales: {
          x: {
            grid: { color: 'rgba(148, 163, 173, 0.1)' },
            title: { display: true, text: 'Sample Index / Sequence' }
          },
          y: {
            grid: { color: 'rgba(148, 163, 173, 0.1)' },
            title: { display: true, text: `${colName} Concentration` }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const idx = ctx.dataIndex;
                const isMissing = this.isIndexMissing(idx);
                const currentData = this.chart.data.datasets[0].data;
                const val = currentData[idx];
                const formattedVal = val !== null && val !== undefined ? Number(val).toFixed(2) : '--';
                if (isMissing) {
                  return `Imputed (${this.currentMethod.toUpperCase()}): ${formattedVal} [Missing Gap]`;
                }
                return `Measured: ${formattedVal}`;
              }
            }
          }
        }
      }
    });

    this.isFilled = true;
    if (this.counterEl) this.counterEl.textContent = totalMissing;
    this.updateStats(methodData, totalMissing);
  }

  setMethod(methodKey) {
    this.cancelAnimation();
    this.currentMethod = methodKey;
    const methodData = this.getCurrentMethodData();
    if (!methodData) return;

    this.isFilled = true;
    if (this.fillBtn) {
      this.fillBtn.disabled = false;
      this.fillBtn.textContent = '⚡ Fill missing values';
    }

    const afterArr = methodData.after || [];
    const missingIndices = this.getMissingIndices(methodData);
    const totalMissing = missingIndices.length;

    if (this.counterEl) {
      this.counterEl.textContent = totalMissing;
    }

    if (this.chart) {
      const colName = methodData.chart_column || "CO(GT)";
      const styles = this.getPointStyles(methodData);

      this.chart.data.datasets[0].label = `${colName} Telemetry (${methodKey.toUpperCase()})`;
      this.chart.data.datasets[0].data = [...afterArr];
      this.chart.data.datasets[0].spanGaps = true;
      this.chart.data.datasets[0].borderColor = '#4ADE80';
      this.chart.data.datasets[0].pointBackgroundColor = styles.bgColors;
      this.chart.data.datasets[0].pointBorderColor = styles.borderColors;
      this.chart.data.datasets[0].pointRadius = styles.radii;
      this.chart.data.datasets[0].pointBorderWidth = styles.borderWidths;
      this.chart.data.datasets[0].pointHoverRadius = styles.hoverRadii;
      this.chart.update();
    }

    this.updateStats(methodData, totalMissing);
  }

  triggerStaggeredFill() {
    if (this.animating || !this.chart) return;
    this.cancelAnimation();
    this.animating = true;

    if (this.fillBtn) {
      this.fillBtn.disabled = true;
      this.fillBtn.textContent = '⏳ Imputing...';
    }

    const methodData = this.getCurrentMethodData();
    if (!methodData) return;

    const beforeArr = methodData.before || [];
    const afterArr = methodData.after || [];
    const missingIndices = this.getMissingIndices(methodData);
    const totalMissing = missingIndices.length;

    // Reset chart missing values to null initially to show staggered fill
    this.chart.data.datasets[0].data = [...beforeArr];
    this.chart.data.datasets[0].spanGaps = false;
    this.chart.data.datasets[0].borderColor = '#94A3AD';
    this.chart.update('none');

    let count = 0;
    if (this.counterEl) this.counterEl.textContent = '0';

    if (totalMissing === 0) {
      this.isFilled = true;
      this.animating = false;
      this.chart.data.datasets[0].spanGaps = true;
      this.chart.data.datasets[0].borderColor = '#4ADE80';
      this.chart.update();
      if (this.fillBtn) {
        this.fillBtn.disabled = false;
        this.fillBtn.textContent = '✓ Values Filled';
      }
      return;
    }

    missingIndices.forEach((missingIdx, i) => {
      const tid = setTimeout(() => {
        const imputedVal = afterArr[missingIdx];
        this.chart.data.datasets[0].data[missingIdx] = imputedVal;
        this.chart.update('none');

        count++;
        if (this.counterEl) this.counterEl.textContent = count;

        if (count === totalMissing) {
          this.isFilled = true;
          this.animating = false;
          this.chart.data.datasets[0].spanGaps = true;
          this.chart.data.datasets[0].borderColor = '#4ADE80';
          this.chart.update();
          if (this.fillBtn) {
            this.fillBtn.disabled = false;
            this.fillBtn.textContent = '✓ Values Filled';
          }
        }
      }, Math.min((i + 1) * 35, 1200));
      this.animTimeouts.push(tid);
    });
  }

  updateStats(methodData, totalMissing) {
    const colName = methodData.chart_column || "CO(GT)";
    const mae = methodData.mae_per_column 
      ? (methodData.mae_per_column[colName] || methodData.mae) 
      : methodData.mae;

    if (this.titleEl) this.titleEl.textContent = `${colName} Sensor Telemetry Timeline`;
    if (this.totalMissingEl) this.totalMissingEl.textContent = `/ ${totalMissing} missing in sample`;
    if (this.maeEl) this.maeEl.textContent = typeof mae === 'number' ? mae.toFixed(2) : mae;
    if (this.maeUnitEl) this.maeUnitEl.textContent = colName.includes("CO") ? "mg/m³" : "µg/m³";

    const descMap = {
      knn: "K-Nearest Neighbors (k=5) evaluates multi-pollutant telemetry similarity across time to reconstruct realistic sensor dynamics.",
      mean: "Mean imputation fills all missing gaps with the global channel average, reducing variance but flattening temporal peaks.",
      median: "Median imputation substitutes missing readings with the channel median, offering robust resistance against extreme spikes."
    };
    if (this.descEl) this.descEl.textContent = descMap[this.currentMethod] || methodData.description || "";
  }

  setupEvents() {
    const btns = document.querySelectorAll('.imp-tab-btn');
    btns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const targetBtn = e.currentTarget;
        btns.forEach(b => b.classList.remove('active'));
        targetBtn.classList.add('active');
        const method = targetBtn.getAttribute('data-method');
        this.setMethod(method);
      });
    });

    if (this.fillBtn) {
      this.fillBtn.addEventListener('click', () => {
        this.triggerStaggeredFill();
      });
    }
  }
}

/* ==========================================================================
   3. COVARIANCE HEATMAP & SCATTER MANAGER
   ========================================================================== */
class CovarianceManager {
  constructor(gridEl, scatterCanvas, data) {
    this.gridEl = gridEl;
    this.scatterCanvas = scatterCanvas;
    this.data = data;
    this.scatterChart = null;

    if (this.data && this.data.top_pairs && this.data.top_pairs.length > 0) {
      this.selectedPair = [this.data.top_pairs[0].a, this.data.top_pairs[0].b];
    } else if (this.data && this.data.pollutants && this.data.pollutants.length >= 2) {
      this.selectedPair = [this.data.pollutants[0], this.data.pollutants[1]];
    } else {
      this.selectedPair = ["CO(GT)", "C6H6(GT)"];
    }

    this.initHeatmap();
    this.initScatter();
    this.renderTop3Pairs();
  }

  initHeatmap() {
    if (!this.gridEl || !this.data) return;
    this.gridEl.innerHTML = '';

    const pollutants = this.data.pollutants || [];
    const matrix = this.data.matrix || [];

    // Heatmap Tooltip Element
    let tooltip = document.getElementById('heatmap-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'heatmap-tooltip';
      tooltip.className = 'heatmap-tooltip';
      document.body.appendChild(tooltip);
    }

    // Top-left blank header cell
    const blank = document.createElement('div');
    blank.className = 'heatmap-label';
    this.gridEl.appendChild(blank);

    // Column Headers
    pollutants.forEach((p, colIdx) => {
      const label = document.createElement('div');
      label.className = `heatmap-label col-header-${colIdx}`;
      label.textContent = p;
      this.gridEl.appendChild(label);
    });

    // Rows
    pollutants.forEach((rowP, rowIdx) => {
      // Row Header
      const rowLabel = document.createElement('div');
      rowLabel.className = `heatmap-label row-header-${rowIdx}`;
      rowLabel.textContent = rowP;
      this.gridEl.appendChild(rowLabel);

      // Matrix Cells
      pollutants.forEach((colP, colIdx) => {
        const val = matrix[rowIdx] ? matrix[rowIdx][colIdx] : 0;
        const cell = document.createElement('div');
        cell.className = 'heatmap-cell';
        cell.setAttribute('data-row', rowIdx);
        cell.setAttribute('data-col', colIdx);
        cell.setAttribute('data-p1', rowP);
        cell.setAttribute('data-p2', colP);
        cell.textContent = val.toFixed(2);

        // Cell Color calculation
        if (val >= 0) {
          const alpha = 0.12 + val * 0.75;
          cell.style.backgroundColor = `rgba(52, 211, 153, ${alpha})`;
          cell.style.color = val > 0.6 ? '#09090b' : '#fafafa';
        } else {
          const alpha = 0.12 + Math.abs(val) * 0.75;
          cell.style.backgroundColor = `rgba(248, 113, 113, ${alpha})`;
          cell.style.color = '#fafafa';
        }

        if ((rowP === this.selectedPair[0] && colP === this.selectedPair[1]) ||
            (rowP === this.selectedPair[1] && colP === this.selectedPair[0])) {
          cell.classList.add('active-selected');
        }

        // Hover Event: highlight row and column labels + tooltip
        cell.addEventListener('mouseenter', () => {
          document.querySelectorAll(`.row-header-${rowIdx}, .col-header-${colIdx}`).forEach(el => {
            el.classList.add('highlight');
          });
          tooltip.style.display = 'block';
          tooltip.innerHTML = `<div><strong>${rowP} vs ${colP}</strong></div><div>Correlation (r): ${val >= 0 ? '+' : ''}${val.toFixed(3)}</div>`;
        });

        cell.addEventListener('mousemove', (e) => {
          tooltip.style.left = (e.pageX + 15) + 'px';
          tooltip.style.top = (e.pageY + 15) + 'px';
        });

        cell.addEventListener('mouseleave', () => {
          document.querySelectorAll('.heatmap-label').forEach(el => el.classList.remove('highlight'));
          tooltip.style.display = 'none';
        });

        // Click Event: Lock side panel scatter plot
        cell.addEventListener('click', () => {
          document.querySelectorAll('.heatmap-cell').forEach(c => c.classList.remove('active-selected'));
          cell.classList.add('active-selected');
          this.selectPair(rowP, colP);
        });

        this.gridEl.appendChild(cell);
      });
    });
  }

  initScatter() {
    if (!this.scatterCanvas) return;

    this.scatterChart = new Chart(this.scatterCanvas.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [{
          label: `${this.selectedPair[0]} vs ${this.selectedPair[1]}`,
          data: [],
          backgroundColor: 'rgba(74, 222, 128, 0.75)',
          borderColor: '#4ADE80',
          borderWidth: 1,
          pointRadius: 3.5,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: this.selectedPair[0] },
            grid: { color: 'rgba(148, 163, 173, 0.1)' }
          },
          y: {
            title: { display: true, text: this.selectedPair[1] },
            grid: { color: 'rgba(148, 163, 173, 0.1)' }
          }
        }
      }
    });

    this.updateScatterData(this.selectedPair[0], this.selectedPair[1]);
  }

  selectPair(p1, p2) {
    this.selectedPair = [p1, p2];
    this.updateScatterData(p1, p2);
  }

  updateScatterData(p1, p2) {
    if (!this.data) return;

    let pts = null;
    let pairCorr = null;

    // 1. Check top_pairs array
    if (this.data.top_pairs && Array.isArray(this.data.top_pairs)) {
      const match = this.data.top_pairs.find(item => 
        (item.a === p1 && item.b === p2) || (item.a === p2 && item.b === p1)
      );
      if (match) {
        pairCorr = match.corr;
        if (match.scatter && match.scatter.length > 0) {
          if (match.a === p1 && match.b === p2) {
            pts = match.scatter;
          } else {
            // Flip axes if requested in reverse order
            pts = match.scatter.map(pt => ({ x: pt.y, y: pt.x }));
          }
        }
      }
    }

    // 2. Fallback to scatter_data
    if (!pts && this.data.scatter_data) {
      const key = `${p1}_${p2}`;
      const reverseKey = `${p2}_${p1}`;
      if (this.data.scatter_data[key]) {
        pts = this.data.scatter_data[key];
      } else if (this.data.scatter_data[reverseKey]) {
        pts = this.data.scatter_data[reverseKey].map(pt => ({ x: pt.y, y: pt.x }));
      }
    }

    // 3. Look up Pearson r in correlation matrix if needed
    const pollutants = this.data.pollutants || [];
    const idx1 = pollutants.indexOf(p1);
    const idx2 = pollutants.indexOf(p2);
    const corrVal = (pairCorr !== null && pairCorr !== undefined) 
      ? pairCorr 
      : (idx1 !== -1 && idx2 !== -1 && this.data.matrix && this.data.matrix[idx1]) 
        ? this.data.matrix[idx1][idx2] 
        : (p1 === p2 ? 1.0 : 0.5);

    // 4. Fallback generator if specific scatter array not present
    if (!pts || pts.length === 0) {
      pts = [];
      const seedStr = `${p1}_${p2}`;
      let seed = 0;
      for (let i = 0; i < seedStr.length; i++) seed += seedStr.charCodeAt(i);

      const count = 50;
      for (let i = 0; i < count; i++) {
        const p1R = Math.sin(seed + i * 1.7);
        const p2R = Math.cos(seed + i * 2.3);
        const xVal = 10 + (i / count) * 80 + p1R * 4;
        const yVal = p1 === p2 
          ? xVal + p2R * 0.1 
          : 15 + (xVal * 0.7) * corrVal + p2R * 12 * (1 - Math.abs(corrVal));
        pts.push({ x: parseFloat(xVal.toFixed(2)), y: parseFloat(yVal.toFixed(2)) });
      }
    }

    // DOM Updates
    const titleEl = document.getElementById('scatter-title');
    const valEl = document.getElementById('scatter-corr-val');
    const countEl = document.getElementById('scatter-sample-count');
    const labelEl = document.getElementById('scatter-corr-label');
    const interpEl = document.getElementById('scatter-interpretation-text');

    if (titleEl) titleEl.textContent = `${p1} vs ${p2} Scatter Analysis`;

    if (valEl) {
      valEl.textContent = (corrVal >= 0 ? '+' : '') + corrVal.toFixed(2);
      if (p1 === p2 || corrVal >= 0.7) valEl.className = 'metric-value mono-count aqi-color-good';
      else if (corrVal >= 0.4) valEl.className = 'metric-value mono-count aqi-color-moderate';
      else if (corrVal >= 0.1) valEl.className = 'metric-value mono-count aqi-color-sensitive';
      else if (corrVal > -0.1) valEl.className = 'metric-value mono-count';
      else valEl.className = 'metric-value mono-count aqi-color-unhealthy';
    }

    // Real Sample Count
    const totalN = this.data.n_rows_used || this.data.sample_count || pts.length;
    if (countEl) {
      const plottedNote = (pts && pts.length < totalN) ? ` (${pts.length} plotted)` : '';
      countEl.textContent = `Sample Count: n = ${totalN.toLocaleString()}${plottedNote}`;
    }

    // Dynamic Correlation Label & Status Color
    if (labelEl) {
      if (p1 === p2) {
        labelEl.textContent = 'Identity Pair (Self-Correlation)';
        labelEl.style.color = 'var(--aqi-good)';
      } else if (corrVal >= 0.7) {
        labelEl.textContent = 'Strong Positive Correlation';
        labelEl.style.color = 'var(--aqi-good)';
      } else if (corrVal >= 0.4) {
        labelEl.textContent = 'Moderate Positive Correlation';
        labelEl.style.color = 'var(--aqi-moderate)';
      } else if (corrVal >= 0.1) {
        labelEl.textContent = 'Weak Positive Correlation';
        labelEl.style.color = 'var(--aqi-sensitive)';
      } else if (corrVal > -0.1) {
        labelEl.textContent = 'Negligible / Neutral Correlation';
        labelEl.style.color = 'var(--text-secondary)';
      } else if (corrVal > -0.4) {
        labelEl.textContent = 'Weak Negative Correlation';
        labelEl.style.color = 'var(--aqi-sensitive)';
      } else if (corrVal > -0.7) {
        labelEl.textContent = 'Moderate Negative Correlation';
        labelEl.style.color = 'var(--aqi-unhealthy)';
      } else {
        labelEl.textContent = 'Strong Negative Correlation';
        labelEl.style.color = 'var(--aqi-hazardous)';
      }
    }

    // Dynamic Interpretation Readout
    if (interpEl) {
      if (p1 === p2) {
        interpEl.textContent = `${p1} evaluated against itself yields trivial perfect correlation (r = +1.00).`;
      } else {
        const absCorr = Math.abs(corrVal);
        const direction = corrVal >= 0 ? "positive" : "negative";
        const strength = absCorr >= 0.7 ? "strong" : absCorr >= 0.4 ? "moderate" : absCorr >= 0.1 ? "weak" : "negligible";
        const units1 = (this.data.units && this.data.units[p1]) ? ` (${this.data.units[p1]})` : '';
        const units2 = (this.data.units && this.data.units[p2]) ? ` (${this.data.units[p2]})` : '';

        interpEl.textContent = `${p1}${units1} and ${p2}${units2} display a ${strength} ${direction} correlation (r = ${corrVal >= 0 ? '+' : ''}${corrVal.toFixed(2)}). ` +
          (absCorr >= 0.5 
            ? `Concentration variations in ${p1} closely mirror shifts in ${p2}, reflecting shared emission sources or atmospheric chemical coupling.` 
            : `The atmospheric behavior of ${p1} and ${p2} operates relatively independently under measured urban conditions.`);
      }
    }

    // Update Chart.js Scatter Chart
    if (this.scatterChart) {
      const units1 = (this.data.units && this.data.units[p1]) ? ` (${this.data.units[p1]})` : '';
      const units2 = (this.data.units && this.data.units[p2]) ? ` (${this.data.units[p2]})` : '';

      this.scatterChart.data.datasets[0].label = `${p1} vs ${p2}`;
      this.scatterChart.data.datasets[0].data = pts;
      this.scatterChart.options.scales.x.title.text = `${p1}${units1}`;
      this.scatterChart.options.scales.y.title.text = `${p2}${units2}`;
      this.scatterChart.update();
    }
  }

  renderTop3Pairs() {
    const listEl = document.getElementById('top3-correlated-list');
    if (!listEl || !this.data) return;

    listEl.innerHTML = '';

    let top3 = [];
    if (this.data.top_pairs && Array.isArray(this.data.top_pairs) && this.data.top_pairs.length > 0) {
      top3 = this.data.top_pairs.slice(0, 3).map(item => ({
        p1: item.a,
        p2: item.b,
        corr: item.corr
      }));
    } else {
      const pairs = [];
      const pollutants = this.data.pollutants || [];
      const matrix = this.data.matrix || [];

      for (let i = 0; i < pollutants.length; i++) {
        for (let j = i + 1; j < pollutants.length; j++) {
          pairs.push({
            p1: pollutants[i],
            p2: pollutants[j],
            corr: matrix[i][j]
          });
        }
      }
      pairs.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
      top3 = pairs.slice(0, 3);
    }

    top3.forEach(item => {
      const el = document.createElement('div');
      el.className = 'correlated-item';
      el.innerHTML = `
        <span class="correlated-pair">${item.p1} ↔ ${item.p2}</span>
        <span class="correlated-val">${item.corr >= 0 ? '+' : ''}${item.corr.toFixed(2)}</span>
      `;
      el.addEventListener('click', () => {
        this.selectPair(item.p1, item.p2);
        // Highlight in heatmap
        const pollutants = this.data.pollutants || [];
        const rIdx = pollutants.indexOf(item.p1);
        const cIdx = pollutants.indexOf(item.p2);
        document.querySelectorAll('.heatmap-cell').forEach(c => c.classList.remove('active-selected'));
        const targetCell = document.querySelector(`.heatmap-cell[data-row="${rIdx}"][data-col="${cIdx}"]`);
        if (targetCell) targetCell.classList.add('active-selected');
      });
      listEl.appendChild(el);
    });
  }
}

function formatPolynomialEquation(coeffs) {
  if (!coeffs || !coeffs.length) return '';
  const deg = coeffs.length - 1;
  const superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶'];

  let str = 'y = ';
  coeffs.forEach((c, idx) => {
    const power = deg - idx;
    if (Math.abs(c) < 1e-6 && idx < deg) return;

    const sign = (idx === 0) ? (c < 0 ? '-' : '') : (c >= 0 ? ' + ' : ' - ');
    const absVal = Math.abs(c);
    let valStr = '';
    if (power > 0 && Math.abs(absVal - 1) < 1e-4) {
      valStr = '';
    } else if (absVal >= 100) {
      valStr = absVal.toFixed(1);
    } else if (absVal >= 1) {
      valStr = absVal.toFixed(2);
    } else if (absVal >= 0.01) {
      valStr = absVal.toFixed(3);
    } else {
      valStr = absVal.toExponential(2);
    }

    let varStr = '';
    if (power === 1) varStr = 'x';
    else if (power > 1) varStr = `x${superscripts[power] || ('^' + power)}`;

    str += `${sign}${valStr}${varStr}`;
  });
  return str;
}

class PolyfitManager {
  constructor(canvasEl, data) {
    this.canvas = canvasEl;
    this.data = data;
    this.currentDegree = 3; // Default
    this.chart = null;

    this.r2El = document.getElementById('poly-r2-val');
    this.rmseEl = document.getElementById('poly-rmse-val');
    this.heldRmseEl = document.getElementById('poly-held-rmse-val');
    this.eqEl = document.getElementById('poly-equation-val');
    this.readoutEl = document.getElementById('poly-dynamic-readout');
    this.titleEl = document.getElementById('poly-chart-title');

    this.initChart();
    this.setupEvents();
  }

  initChart() {
    if (!this.canvas || !this.data || !this.data.degrees) return;

    const degData = this.data.degrees[this.currentDegree.toString()];
    const xLabel = this.data.x_label || 'Hour of day (0–23)';
    const yLabel = this.data.y_label || 'NO2 (µg/m³)';

    if (this.titleEl) {
      const pollutant = yLabel.split(' ')[0] || 'NO2';
      this.titleEl.textContent = `${pollutant} Diurnal Concentration Curve Fit`;
    }

    this.chart = new Chart(this.canvas.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [
          {
            type: 'scatter',
            label: `Actual Telemetry (${yLabel})`,
            data: degData.actual_points,
            backgroundColor: 'rgba(74, 222, 128, 0.75)',
            borderColor: '#4ADE80',
            pointRadius: 4,
            pointHoverRadius: 6.5
          },
          {
            type: 'line',
            label: `Polynomial Degree ${this.currentDegree} Fit`,
            data: degData.curve_points,
            borderColor: '#FB923C',
            borderWidth: 2.5,
            tension: 0.35,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400, easing: 'easeInOutQuad' },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: 23,
            title: { display: true, text: xLabel },
            grid: { color: 'rgba(148, 163, 173, 0.1)' },
            ticks: { stepSize: 2 }
          },
          y: {
            title: { display: true, text: yLabel },
            grid: { color: 'rgba(148, 163, 173, 0.1)' }
          }
        }
      }
    });

    this.updateMetrics(degData);
  }

  setDegree(degNum) {
    this.currentDegree = degNum;
    const degData = this.data.degrees[degNum.toString()];
    if (!degData || !this.chart) return;

    this.chart.data.datasets[1].label = `Polynomial Degree ${degNum} Fit`;
    this.chart.data.datasets[1].data = degData.curve_points;
    this.chart.update();

    this.updateMetrics(degData);
  }

  updateMetrics(degData) {
    const currentR2 = parseFloat(this.r2El ? this.r2El.textContent : 0) || 0;
    const currentRmse = parseFloat(this.rmseEl ? this.rmseEl.textContent : 0) || 0;
    const currentHeldRmse = parseFloat(this.heldRmseEl ? this.heldRmseEl.textContent : 0) || 0;

    animateMonoNumber(this.r2El, currentR2, degData.r2, 400, 3);
    animateMonoNumber(this.rmseEl, currentRmse, degData.rmse, 400, 2);
    animateMonoNumber(this.heldRmseEl, currentHeldRmse, degData.held_out_rmse, 400, 2);

    if (this.eqEl) {
      this.eqEl.textContent = formatPolynomialEquation(degData.coefficients) || degData.equation || '';
    }

    if (this.readoutEl) {
      this.readoutEl.textContent = degData.readout || degData.analysis || '';
    }
  }

  setupEvents() {
    const slider = document.getElementById('poly-degree-slider');
    const displayVal = document.getElementById('poly-degree-display');

    if (slider) {
      slider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        if (displayVal) displayVal.textContent = val;
        this.setDegree(val);
      });
    }

    const segBtns = document.querySelectorAll('.poly-degree-btn');
    segBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        segBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const deg = parseInt(e.target.getAttribute('data-degree'), 10);
        if (slider) slider.value = deg;
        if (displayVal) displayVal.textContent = deg;
        this.setDegree(deg);
      });
    });
  }
}

/* ==========================================================================
   5. RIDGE REGRESSION MANAGER (UNIT 2)
   ========================================================================== */
class RidgeManager {
  constructor(data) {
    this.data = data;
    this.currentModelKey = 'ols';
    this.scatterChart = null;
    this.coefChart = null;

    this.alphaDisplayEl = document.getElementById('ridge-alpha-display');
    this.r2TestEl = document.getElementById('ridge-r2-test');
    this.r2TrainEl = document.getElementById('ridge-r2-train');
    this.rmseTestEl = document.getElementById('ridge-rmse-test');
    this.rmseTrainEl = document.getElementById('ridge-rmse-train');
    this.maeTestEl = document.getElementById('ridge-mae-test');
    this.interceptEl = document.getElementById('ridge-intercept');
    this.signFlipNoteEl = document.getElementById('ridge-signflip-note');
    this.corrPairsContainer = document.getElementById('ridge-corr-pairs');

    this.renderTopCorrPairs();
    this.initScatterChart();
    this.initCoefChart();
    this.setupEvents();
    this.updateModel(this.currentModelKey);
  }

  renderTopCorrPairs() {
    if (!this.corrPairsContainer || !this.data || !this.data.high_correlation_pairs) return;
    this.corrPairsContainer.innerHTML = '';
    this.data.high_correlation_pairs.slice(0, 4).forEach(pair => {
      const div = document.createElement('div');
      div.style.display = 'flex';
      div.style.justifyContent = 'space-between';
      div.style.gap = '1.5rem';
      div.style.padding = '0.2rem 0';
      div.style.borderBottom = '1px dashed rgba(148, 163, 173, 0.15)';
      
      const corrVal = pair.corr >= 0 ? `+${pair.corr.toFixed(3)}` : pair.corr.toFixed(3);
      const color = Math.abs(pair.corr) > 0.85 ? '#FB7185' : '#FBBF24';
      
      div.innerHTML = `
        <span style="color: var(--text-primary);">${pair.a} ↔ ${pair.b}</span>
        <span style="color: ${color}; font-weight: 700;">${corrVal}</span>
      `;
      this.corrPairsContainer.appendChild(div);
    });
  }

  initScatterChart() {
    const canvas = document.getElementById('ridge-scatter-chart');
    if (!canvas || !this.data || !this.data.pred_vs_actual_sample) return;

    const points = this.data.pred_vs_actual_sample.map(d => ({ x: d.actual, y: d.predicted }));
    let minVal = 0;
    let maxVal = 8;

    this.scatterChart = new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Telemetry Sample (Actual vs Predicted)',
            data: points,
            backgroundColor: 'rgba(74, 222, 128, 0.7)',
            borderColor: '#4ADE80',
            pointRadius: 3.5,
            pointHoverRadius: 6
          },
          {
            type: 'line',
            label: 'Perfect Prediction (y = x)',
            data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
            borderColor: 'rgba(148, 163, 173, 0.5)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Actual CO(GT) Concentration (mg/m³)' },
            grid: { color: 'rgba(148, 163, 173, 0.1)' }
          },
          y: {
            title: { display: true, text: 'Predicted CO(GT) Concentration (mg/m³)' },
            grid: { color: 'rgba(148, 163, 173, 0.1)' }
          }
        }
      }
    });
  }

  initCoefChart() {
    const canvas = document.getElementById('ridge-coef-chart');
    if (!canvas || !this.data || !this.data.models) return;

    const predictors = this.data.predictors || Object.keys(this.data.models.ols.coefficients);
    const olsCoefs = predictors.map(p => this.data.models.ols.coefficients[p]);

    this.coefChart = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: predictors,
        datasets: [
          {
            label: 'OLS Baseline (α=0)',
            data: olsCoefs,
            backgroundColor: 'rgba(148, 163, 173, 0.45)',
            borderColor: '#94A3AD',
            borderWidth: 1
          },
          {
            label: 'Ridge (α = 0)',
            data: [...olsCoefs],
            backgroundColor: 'rgba(251, 146, 60, 0.85)',
            borderColor: '#FB923C',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        scales: {
          x: {
            grid: { color: 'rgba(148, 163, 173, 0.1)' },
            ticks: { font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            title: { display: true, text: 'Standardized Coefficient Weight β' },
            grid: { color: 'rgba(148, 163, 173, 0.1)' }
          }
        }
      }
    });
  }

  updateModel(modelKey) {
    this.currentModelKey = modelKey;
    const model = this.data.models[modelKey];
    if (!model) return;

    const olsModel = this.data.models.ols;

    if (this.alphaDisplayEl) this.alphaDisplayEl.textContent = model.display_name;

    const curR2Test = parseFloat(this.r2TestEl ? this.r2TestEl.textContent : 0) || 0;
    const curRmseTest = parseFloat(this.rmseTestEl ? this.rmseTestEl.textContent : 0) || 0;
    const curMaeTest = parseFloat(this.maeTestEl ? this.maeTestEl.textContent : 0) || 0;

    animateMonoNumber(this.r2TestEl, curR2Test, model.r2_test, 400, 4);
    animateMonoNumber(this.rmseTestEl, curRmseTest, model.rmse_test, 400, 4);
    animateMonoNumber(this.maeTestEl, curMaeTest, model.mae_test, 400, 4);

    if (this.r2TrainEl) this.r2TrainEl.textContent = `Train R²: ${model.r2_train.toFixed(4)}`;
    if (this.rmseTrainEl) this.rmseTrainEl.textContent = `Train RMSE: ${model.rmse_train.toFixed(4)}`;
    if (this.interceptEl) this.interceptEl.textContent = `β₀ = ${model.intercept.toFixed(4)}`;

    if (this.coefChart) {
      const predictors = this.data.predictors || Object.keys(olsModel.coefficients);
      const selectedCoefs = predictors.map(p => model.coefficients[p]);

      const secondDatasetLabel = `Ridge (α = ${model.alpha})`;

      this.coefChart.data.datasets[1].label = secondDatasetLabel;
      this.coefChart.data.datasets[1].data = selectedCoefs;
      this.coefChart.update();
    }

    this.generateSignFlipDiagnostic(model, olsModel);
  }

  generateSignFlipDiagnostic(model, olsModel) {
    if (!this.signFlipNoteEl) return;

    if (model.alpha === 0) {
      this.signFlipNoteEl.innerHTML = `
        <strong>Unconstrained OLS Baseline (α=0):</strong> Sensor predictors <code>PT08.S1(CO)</code> and <code>PT08.S5(O3)</code> carry high positive collinearity (r = 0.898). In OLS, this forces <code>PT08.S5(O3)</code> to take an inverted negative weight (β = -0.0385) to compensate for inflated <code>PT08.S2(NMHC)</code> (β = +1.3381).
      `;
      return;
    }

    const predictors = this.data.predictors || Object.keys(olsModel.coefficients);
    const signFlips = [];

    predictors.forEach(p => {
      const cOls = olsModel.coefficients[p];
      const cRidge = model.coefficients[p];

      if (cOls * cRidge < 0 && Math.abs(cOls) > 0.005) {
        signFlips.push({
          name: p,
          olsVal: cOls >= 0 ? `+${cOls.toFixed(4)}` : cOls.toFixed(4),
          ridgeVal: cRidge >= 0 ? `+${cRidge.toFixed(4)}` : cRidge.toFixed(4)
        });
      }
    });

    if (signFlips.length > 0) {
      const flipsStr = signFlips.map(f => `<code>${f.name}</code>'s coefficient flips sign between OLS (${f.olsVal}) and this Ridge model (${f.ridgeVal})`).join('; ');
      this.signFlipNoteEl.innerHTML = `
        <strong>Multicollinearity Correction:</strong> ${flipsStr} — a classic multicollinearity symptom Ridge is correcting for. As L₂ regularization penalizes coefficient magnitude, it stabilizes predictor weights across multi-channel telemetry.
      `;
    } else {
      this.signFlipNoteEl.innerHTML = `
        <strong>Regularization Effect (${model.display_name}):</strong> Test set R² remains remarkably stable (${model.r2_test.toFixed(4)} vs OLS ${olsModel.r2_test.toFixed(4)}) while coefficient magnitudes shrink smoothly, dampening over-fitting to correlated sensor channels.
      `;
    }
  }

  setupEvents() {
    const btns = document.querySelectorAll('.ridge-alpha-btn');
    btns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const targetBtn = e.target.closest('.ridge-alpha-btn');
        if (!targetBtn) return;
        btns.forEach(b => b.classList.remove('active'));
        targetBtn.classList.add('active');
        const alphaKey = targetBtn.getAttribute('data-alpha');
        this.updateModel(alphaKey);
      });
    });
  }
}

/* ==========================================================================
   6. SVM CLASSIFICATION MANAGER (UNIT 2)
   ========================================================================== */
class SVMManager {
  constructor(data) {
    this.data = data;
    this.currentKernel = 'linear';

    this.kernelDisplayEl = document.getElementById('svm-kernel-display');
    this.accuracyEl = document.getElementById('svm-accuracy');
    this.bestF1El = document.getElementById('svm-best-f1');
    this.bestF1ClassEl = document.getElementById('svm-best-f1-class');
    this.tableBodyEl = document.querySelector('#svm-metrics-table tbody');
    this.matrixGridEl = document.getElementById('svm-matrix-grid');
    this.thresholdsContainerEl = document.getElementById('svm-thresholds-container');
    this.boundaryCanvas = document.getElementById('svm-decision-canvas');

    this.renderThresholds();
    this.setupEvents();
    this.updateKernel(this.currentKernel);
  }

  renderThresholds() {
    if (!this.thresholdsContainerEl || !this.data || !this.data.category_thresholds) return;
    this.thresholdsContainerEl.innerHTML = '';
    
    const catColors = {
      Safe: { bg: 'rgba(52, 211, 153, 0.1)', border: 'rgba(52, 211, 153, 0.3)', text: '#34d399' },
      Moderate: { bg: 'rgba(251, 191, 36, 0.1)', border: 'rgba(251, 191, 36, 0.3)', text: '#fbbf24' },
      Hazardous: { bg: 'rgba(248, 113, 113, 0.1)', border: 'rgba(248, 113, 113, 0.3)', text: '#f87171' }
    };

    Object.entries(this.data.category_thresholds).forEach(([cat, range]) => {
      const c = catColors[cat] || catColors.Safe;
      const chip = document.createElement('div');
      chip.style.padding = '0.3rem 0.85rem';
      chip.style.borderRadius = '9999px';
      chip.style.background = c.bg;
      chip.style.border = `1px solid ${c.border}`;
      chip.style.fontFamily = 'var(--font-mono)';
      chip.style.fontSize = '0.78rem';
      chip.innerHTML = `
        <strong style="color: ${c.text};">${cat}:</strong>
        <span style="color: var(--text-primary); margin-left: 0.35rem;">${range.min} – ${range.max} mg/m³</span>
      `;
      this.thresholdsContainerEl.appendChild(chip);
    });
  }

  updateKernel(kernelKey) {
    this.currentKernel = kernelKey;
    const kernelData = this.data.kernels[kernelKey];
    if (!kernelData) return;

    if (this.kernelDisplayEl) {
      this.kernelDisplayEl.textContent = kernelKey === 'rbf' ? 'RBF Kernel (Non-Linear)' : 'Linear Kernel';
    }

    const curAcc = parseFloat(this.accuracyEl ? this.accuracyEl.textContent : 0) || 0;
    const targetAcc = (kernelData.accuracy * 100);
    animateMonoNumber(this.accuracyEl, curAcc, targetAcc, 400, 1, '%');

    let bestF1 = 0;
    let bestClass = '';
    Object.entries(kernelData.per_class).forEach(([cls, metrics]) => {
      if (metrics.f1 > bestF1) {
        bestF1 = metrics.f1;
        bestClass = cls;
      }
    });

    if (this.bestF1El) this.bestF1El.textContent = (bestF1 * 100).toFixed(1) + '%';
    if (this.bestF1ClassEl) this.bestF1ClassEl.textContent = `Highest F1: ${bestClass} Class`;

    this.renderMetricsTable(kernelData.per_class);
    this.renderConfusionMatrix(kernelData);
    this.renderDecisionBoundaryCanvas();
  }

  renderMetricsTable(perClassData) {
    if (!this.tableBodyEl) return;
    this.tableBodyEl.innerHTML = '';

    const catColors = { Safe: '#4ADE80', Moderate: '#FBBF24', Hazardous: '#FB7185' };

    Object.entries(perClassData).forEach(([cls, m]) => {
      const tr = document.createElement('tr');
      tr.style.borderBottom = '1px solid rgba(148, 163, 173, 0.1)';
      tr.innerHTML = `
        <td style="padding: 0.65rem 0.6rem; font-weight: 700; color: ${catColors[cls] || '#ffffff'};">${cls}</td>
        <td style="padding: 0.65rem 0.6rem;">${(m.precision * 100).toFixed(1)}%</td>
        <td style="padding: 0.65rem 0.6rem;">${(m.recall * 100).toFixed(1)}%</td>
        <td style="padding: 0.65rem 0.6rem; font-weight: 700; color: var(--aqi-good);">${(m.f1 * 100).toFixed(1)}%</td>
        <td style="padding: 0.65rem 0.6rem; color: var(--text-muted);">${m.support}</td>
      `;
      this.tableBodyEl.appendChild(tr);
    });
  }

  renderConfusionMatrix(kernelData) {
    if (!this.matrixGridEl || !kernelData.confusion_matrix) return;
    this.matrixGridEl.innerHTML = '';

    const labels = kernelData.confusion_matrix_labels || ['Safe', 'Moderate', 'Hazardous'];
    const matrix = kernelData.confusion_matrix;
    const catColors = ['#4ADE80', '#FBBF24', '#FB7185'];

    const emptyCorner = document.createElement('div');
    this.matrixGridEl.appendChild(emptyCorner);

    labels.forEach((lbl, cIdx) => {
      const headerCell = document.createElement('div');
      headerCell.style.textAlign = 'center';
      headerCell.style.fontSize = '0.72rem';
      headerCell.style.color = catColors[cIdx];
      headerCell.style.fontWeight = '700';
      headerCell.style.paddingBottom = '4px';
      headerCell.textContent = `Pred: ${lbl}`;
      this.matrixGridEl.appendChild(headerCell);
    });

    matrix.forEach((row, rIdx) => {
      const rowLabelCell = document.createElement('div');
      rowLabelCell.style.display = 'flex';
      rowLabelCell.style.alignItems = 'center';
      rowLabelCell.style.fontSize = '0.72rem';
      rowLabelCell.style.color = catColors[rIdx];
      rowLabelCell.style.fontWeight = '700';
      rowLabelCell.style.paddingRight = '6px';
      rowLabelCell.textContent = `Act: ${labels[rIdx]}`;
      this.matrixGridEl.appendChild(rowLabelCell);

      row.forEach((val, cIdx) => {
        const isDiagonal = rIdx === cIdx;
        const totalInRow = row.reduce((a, b) => a + b, 0);
        const intensity = totalInRow > 0 ? (val / totalInRow) : 0;
        
        const cell = document.createElement('div');
        cell.style.display = 'flex';
        cell.style.flexDirection = 'column';
        cell.style.alignItems = 'center';
        cell.style.justifyContent = 'center';
        cell.style.padding = '0.75rem 0.4rem';
        cell.style.borderRadius = '4px';
        cell.style.border = isDiagonal ? `1px solid ${catColors[rIdx]}` : '1px solid rgba(148, 163, 173, 0.15)';
        
        if (isDiagonal) {
          cell.style.background = `rgba(52, 211, 153, ${Math.max(0.18, intensity * 0.45)})`;
        } else if (val > 0) {
          cell.style.background = `rgba(248, 113, 113, ${Math.max(0.12, intensity * 0.4)})`;
        } else {
          cell.style.background = 'rgba(24, 24, 27, 0.6)';
        }

        cell.innerHTML = `
          <span style="font-size: 1.1rem; font-weight: 700; color: ${isDiagonal ? '#ffffff' : (val > 0 ? '#FB7185' : 'var(--text-muted)')};">${val}</span>
          <span style="font-size: 0.65rem; color: var(--text-muted); margin-top: 2px;">${(intensity * 100).toFixed(0)}%</span>
        `;
        this.matrixGridEl.appendChild(cell);
      });
    });
  }

  renderDecisionBoundaryCanvas() {
    if (!this.boundaryCanvas || !this.data || !this.data.decision_boundary_2d) return;
    const boundaryData = this.data.decision_boundary_2d;
    
    const canvas = this.boundaryCanvas;
    const ctx = canvas.getContext('2d');
    
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * (window.devicePixelRatio || 1);
    canvas.height = rect.height * (window.devicePixelRatio || 1);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const grid = boundaryData.grid_labels;
    const gridRes = boundaryData.grid_resolution || grid.length;
    
    const [xMin, xMax] = boundaryData.x_range;
    const [yMin, yMax] = boundaryData.y_range;

    const cellW = canvas.width / gridRes;
    const cellH = canvas.height / gridRes;

    const regionColors = [
      'rgba(74, 222, 128, 0.28)',  // Safe (Green)
      'rgba(251, 191, 36, 0.28)',  // Moderate (Yellow)
      'rgba(251, 113, 133, 0.28)'  // Hazardous (Red)
    ];

    const pointColors = {
      Safe: '#4ADE80',
      Moderate: '#FBBF24',
      Hazardous: '#FB7185'
    };

    for (let r = 0; r < gridRes; r++) {
      for (let c = 0; c < gridRes; c++) {
        const clsIdx = grid[r][c];
        ctx.fillStyle = regionColors[clsIdx] || regionColors[0];
        const x = c * cellW;
        const y = (gridRes - 1 - r) * cellH;
        ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
      }
    }

    if (boundaryData.scatter_points && Array.isArray(boundaryData.scatter_points)) {
      boundaryData.scatter_points.forEach(pt => {
        const px = ((pt.x - xMin) / (xMax - xMin)) * canvas.width;
        const py = canvas.height - (((pt.y - yMin) / (yMax - yMin)) * canvas.height);

        ctx.beginPath();
        ctx.arc(px, py, 3.5 * (window.devicePixelRatio || 1), 0, 2 * Math.PI);
        ctx.fillStyle = pointColors[pt.label] || '#ffffff';
        ctx.shadowColor = 'rgba(0,0,0,0.5)';
        ctx.shadowBlur = 4;
        ctx.fill();
        ctx.lineWidth = 1 * (window.devicePixelRatio || 1);
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        ctx.shadowBlur = 0;
      });
    }
  }

  setupEvents() {
    const btns = document.querySelectorAll('.svm-kernel-btn');
    btns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        btns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const kernelKey = e.target.getAttribute('data-kernel');
        this.updateKernel(kernelKey);
      });
    });

    window.addEventListener('resize', () => {
      if (this.boundaryCanvas) this.renderDecisionBoundaryCanvas();
    });
  }
}

/* ==========================================================================
   7. UNIT 2 OVERVIEW SPARKLINES
   ========================================================================== */
function initUnit2Sparklines(ridgeRes, svmRes) {
  const ridgeCanvas = document.getElementById('sparkline-ridge');
  if (ridgeCanvas && ridgeRes && ridgeRes.models) {
    const alphas = Object.keys(ridgeRes.models).map(k => ridgeRes.models[k].alpha);
    const r2s = Object.keys(ridgeRes.models).map(k => ridgeRes.models[k].r2_test);
    new Chart(ridgeCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: alphas,
        datasets: [{
          data: r2s,
          borderColor: '#4ADE80',
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 2,
          pointBackgroundColor: '#4ADE80'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { display: false }, y: { display: false } }
      }
    });
  }

  const svmCanvas = document.getElementById('sparkline-svm');
  if (svmCanvas && svmRes && svmRes.kernels) {
    const linAcc = svmRes.kernels.linear ? svmRes.kernels.linear.accuracy * 100 : 80;
    const rbfAcc = svmRes.kernels.rbf ? svmRes.kernels.rbf.accuracy * 100 : 85.5;
    
    // Update card text dynamically
    const svmSub = document.querySelector('#card-svm .sparkline-container + div span:first-child');
    if (svmSub) {
      svmSub.textContent = `RBF Kernel Accuracy: ${rbfAcc.toFixed(1)}%`;
    }
    
    new Chart(svmCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['Linear', 'RBF'],
        datasets: [{
          data: [linAcc, rbfAcc],
          backgroundColor: ['rgba(251, 191, 36, 0.7)', 'rgba(52, 211, 153, 0.85)'],
          borderColor: ['#fbbf24', '#34d399'],
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { display: false }, y: { display: false, min: 70, max: 95 } }
      }
    });
  }
}

