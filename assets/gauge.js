/**
 * AQI Gauge Engine
 * Real-time animated AQI circular dial/gauge supporting hero display and nav chip
 */

class AQIGauge {
  /**
   * @param {Object} config
   * @param {HTMLCanvasElement} config.canvas
   * @param {HTMLElement} [config.valueEl]
   * @param {HTMLElement} [config.labelEl]
   * @param {boolean} [config.isMini=false]
   */
  constructor(config) {
    this.canvas = config.canvas;
    this.ctx = this.canvas.getContext('2d');
    this.valueEl = config.valueEl;
    this.labelEl = config.labelEl;
    this.isMini = config.isMini || false;

    this.minVal = 0;
    this.maxVal = 300;
    this.currentVal = 40;

    // AQI Breakpoints & Colors
    this.scales = [
      { max: 50, label: 'Good', color: '#4ADE80', class: 'aqi-color-good' },
      { max: 100, label: 'Moderate', color: '#FBBF24', class: 'aqi-color-moderate' },
      { max: 150, label: 'Sensitive', color: '#FB923C', class: 'aqi-color-sensitive' },
      { max: 200, label: 'Unhealthy', color: '#FB7185', class: 'aqi-color-unhealthy' },
      { max: Infinity, label: 'Hazardous', color: '#C084FC', class: 'aqi-color-hazardous' }
    ];

    this.animating = false;
    this.startTime = null;
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.init();
  }

  static getAQIInfo(val) {
    if (val <= 50) return { label: 'Good', color: '#4ADE80', class: 'aqi-color-good' };
    if (val <= 100) return { label: 'Moderate', color: '#FBBF24', class: 'aqi-color-moderate' };
    if (val <= 150) return { label: 'Unhealthy for Sensitive', shortLabel: 'Sensitive', color: '#FB923C', class: 'aqi-color-sensitive' };
    if (val <= 200) return { label: 'Unhealthy', color: '#FB7185', class: 'aqi-color-unhealthy' };
    return { label: 'Hazardous', color: '#C084FC', class: 'aqi-color-hazardous' };
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    if (!this.reducedMotion) {
      this.startLoop();
    } else {
      this.setValue(68);
    }
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || (this.isMini ? 24 : 220);
    this.height = rect.height || (this.isMini ? 24 : 220);

    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.draw(this.currentVal);
  }

  setValue(val) {
    this.currentVal = Math.round(val);
    const info = AQIGauge.getAQIInfo(this.currentVal);

    if (this.valueEl) {
      this.valueEl.textContent = this.currentVal;
      this.valueEl.style.color = info.color;
    }

    if (this.labelEl) {
      this.labelEl.textContent = this.isMini ? (info.shortLabel || info.label) : info.label;
      this.labelEl.style.color = info.color;
    }

    this.draw(this.currentVal);
  }

  startLoop() {
    this.animating = true;
    const animate = (timestamp) => {
      if (!this.startTime) this.startTime = timestamp;
      const elapsed = timestamp - this.startTime;

      // 8-second cycle sweep between 40 and 185
      const period = 8000;
      const sinFactor = (Math.sin((elapsed / period) * 2 * Math.PI - Math.PI / 2) + 1) / 2; // 0 to 1
      const val = 40 + sinFactor * (185 - 40);

      this.setValue(val);
      if (this.animating) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);
  }

  stopLoop() {
    this.animating = false;
  }

  draw(val) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    const centerX = w / 2;
    const centerY = h / 2;

    if (this.isMini) {
      // Mini Nav Chip Gauge
      const radius = Math.min(w, h) / 2 - 2;
      const startAngle = 0.75 * Math.PI;
      const endAngle = 2.25 * Math.PI;
      const pct = Math.min(Math.max((val - 0) / 250, 0), 1);
      const valAngle = startAngle + pct * (endAngle - startAngle);
      const info = AQIGauge.getAQIInfo(val);

      // Track arc
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, startAngle, endAngle);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.stroke();

      // Active value arc
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, startAngle, valAngle);
      ctx.lineWidth = 3;
      ctx.strokeStyle = info.color;
      ctx.stroke();
      return;
    }

    // Full Hero Instrument Gauge
    const radius = Math.min(w, h) / 2 - 24;
    const startAngle = 0.8 * Math.PI;
    const endAngle = 2.2 * Math.PI;
    const totalAngle = endAngle - startAngle;

    // 1. Draw outer background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.lineWidth = 12;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineCap = 'round';
    ctx.stroke();

    // 2. Draw segmented color scale arc
    const segments = [
      { max: 50, color: '#4ADE80' },
      { max: 100, color: '#FBBF24' },
      { max: 150, color: '#FB923C' },
      { max: 200, color: '#FB7185' },
      { max: 300, color: '#C084FC' }
    ];

    let prevMax = 0;
    segments.forEach(seg => {
      const segStartPct = prevMax / 300;
      const segEndPct = seg.max / 300;

      const a1 = startAngle + segStartPct * totalAngle;
      const a2 = startAngle + segEndPct * totalAngle;

      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, a1, a2);
      ctx.lineWidth = 6;
      ctx.strokeStyle = seg.color;
      ctx.stroke();

      prevMax = seg.max;
    });

    // 3. Draw active fill glow arc
    const valPct = Math.min(Math.max(val / 300, 0), 1);
    const valAngle = startAngle + valPct * totalAngle;
    const currentInfo = AQIGauge.getAQIInfo(val);

    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, valAngle);
    ctx.lineWidth = 12;
    ctx.strokeStyle = currentInfo.color;
    ctx.lineCap = 'round';
    ctx.stroke();

    // 4. Draw Needle / Indicator Line
    const needleLen = radius - 18;
    const needleX = centerX + Math.cos(valAngle) * needleLen;
    const needleY = centerY + Math.sin(valAngle) * needleLen;

    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(needleX, needleY);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = currentInfo.color;
    ctx.stroke();

    // Center pivot point
    ctx.beginPath();
    ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
    ctx.fillStyle = currentInfo.color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#09090b';
    ctx.stroke();

    // 5. Dial Ticks & Scale Markings
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#a1a1aa';
    ctx.textAlign = 'center';

    [0, 50, 100, 150, 200, 300].forEach(markVal => {
      const markPct = markVal / 300;
      const markAngle = startAngle + markPct * totalAngle;
      const markR = radius + 14;
      const tx = centerX + Math.cos(markAngle) * markR;
      const ty = centerY + Math.sin(markAngle) * markR + 3;
      ctx.fillText(markVal, tx, ty);
    });
  }
}

// Global Nav AQI Chip Auto-initializer
document.addEventListener('DOMContentLoaded', () => {
  const navCanvas = document.getElementById('nav-aqi-canvas');
  const navVal = document.getElementById('nav-aqi-val');
  const navLabel = document.getElementById('nav-aqi-label');

  if (navCanvas) {
    window.navAQIGauge = new AQIGauge({
      canvas: navCanvas,
      valueEl: navVal,
      labelEl: navLabel,
      isMini: true
    });
  }
});
