(() => {
  'use strict';

  // Global Error Handler for better debugging
  window.onerror = function (msg, url, line) {
    console.error("RiskIQ Debug:", msg, "at", url, ":", line);
    return false;
  };

  // ── CONSTANTS & STATE ──────────────────────────────────────────
  const PAGES = ['dashboard', 'portfolio', 'analysis', 'reports', 'model'];
  let currentState = {
    modelInfo: null,
    history: [],
    stats: { total: 0, high_risk: 0, low_risk: 0, default_rate: 0, avg_confidence: 0 }
  };

  // ── DOM ELEMENTS ───────────────────────────────────────────────
  const navLinks = document.querySelectorAll('.sb-link[data-page]');
  const pageTitle = document.getElementById('page-title');
  const sections = PAGES.reduce((acc, p) => {
    acc[p] = document.getElementById(`page-${p}`);
    return acc;
  }, {});

  // Analysis Form Elements
  const form = document.getElementById('frm');
  const btn = document.getElementById('run-btn');
  const errBox = document.getElementById('err');
  const errTxt = document.getElementById('err-t');
  const idle = document.getElementById('idle');
  const rc = document.getElementById('rc');
  const sc = document.getElementById('sc');
  const dFg = document.getElementById('d-fg');
  const dPct = document.getElementById('d-pct');
  const CIRC = 2 * Math.PI * 35; // r=35

  if (dFg) {
    dFg.style.strokeDasharray = CIRC;
    dFg.style.strokeDashoffset = CIRC;
  }

  // ── NAVIGATION ────────────────────────────────────────────────
  function navigate(pageId) {
    if (!PAGES.includes(pageId)) return;

    // Toggle sections
    PAGES.forEach(p => {
      if (sections[p]) sections[p].style.display = (p === pageId) ? 'block' : 'none';
    });

    // Update Sidebar
    navLinks.forEach(l => {
      l.classList.toggle('active', l.getAttribute('data-page') === pageId);
    });

    // Update Header Title
    const titles = {
      dashboard: 'Risk Assessment Overview',
      portfolio: 'Loan Portfolio History',
      analysis: 'Risk Analysis Engine',
      reports: 'Session Performance Reports',
      model: 'Model Specifications'
    };
    pageTitle.textContent = titles[pageId] || 'RiskIQ Pro';

    // Fetch data for specific pages
    if (pageId === 'dashboard') refreshDashboard();
    if (pageId === 'portfolio') refreshPortfolio();
    if (pageId === 'reports') refreshReports();
    if (pageId === 'model') refreshModelSpecs();

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  navLinks.forEach(link => {
    link.addEventListener('click', () => navigate(link.getAttribute('data-page')));
  });

  // Export globally for inline onclicks
  window.navigate = navigate;

  // ── DATA FETCHING ─────────────────────────────────────────────
  async function apiFetch(url) {
    try {
      const r = await fetch(url);
      return await r.json();
    } catch (e) {
      console.error(`Fetch error [${url}]:`, e);
      return null;
    }
  }

  async function refreshStats() {
    const s = await apiFetch('/api/stats');
    if (s) currentState.stats = s;
  }

  async function refreshHistory() {
    const h = await apiFetch('/api/history');
    if (h) currentState.history = h;
  }

  async function refreshModelInfo() {
    if (currentState.modelInfo) return; // Only fetch once
    const m = await apiFetch('/api/model-info');
    if (m) currentState.modelInfo = m;
  }

  // ── PAGE RENDERING ────────────────────────────────────────────

  async function refreshDashboard() {
    await refreshStats();
    await refreshHistory();
    await refreshModelInfo();

    // KPI Cards
    const kTotal = document.getElementById('kpi-total');
    const kRate = document.getElementById('kpi-rate');
    const kConf = document.getElementById('kpi-conf');
    if (kTotal) kTotal.textContent = (currentState.stats.total || 0).toLocaleString();
    if (kRate) kRate.textContent = (currentState.stats.default_rate || 0) + '%';
    if (kConf) kConf.textContent = (currentState.stats.avg_confidence || 0) + '%';

    // Bars
    const total = currentState.stats.total || 1;
    const bLow = document.getElementById('bar-low-risk');
    const bHigh = document.getElementById('bar-high-risk');
    if (bLow) bLow.style.height = (currentState.stats.low_risk / total * 100) + 'px';
    if (bHigh) bHigh.style.height = (currentState.stats.high_risk / total * 100) + 'px';

    // Top Drivers (from model info)
    const driversBox = document.getElementById('top-drivers');
    if (driversBox && currentState.modelInfo) {
      const top3 = currentState.modelInfo.feature_importance.slice(0, 3);
      driversBox.innerHTML = top3.map(f => `
        <div class="risk-driver-row">
          <div class="rd-top"><span class="rd-name">${f.feature}</span><span class="rd-pct b">${f.pct}%</span></div>
          <div class="rd-track"><div class="rd-fill b" style="width:${f.pct}%"></div></div>
        </div>
      `).join('') + `
        <div class="rd-quote">
          "${top3[0].feature} remains the most influential factor in current assessments."
        </div>
      `;
    }

    // Recent Table
    const tbody = document.getElementById('recent-tbody');
    if (tbody && currentState.history && currentState.history.length) {
      const recent = currentState.history.slice(0, 5);
      tbody.innerHTML = recent.map(r => `
        <tr>
          <td>
            <div style="display:flex;align-items:center;gap:.6rem">
              <div class="tb-avatar" style="width:24px;height:24px;font-size:.6rem">${((r.FullName || 'User').split(' ').map(n => n[0]).join(''))}</div>
              <span style="font-weight:600;font-size:.85rem">${r.FullName || 'Unknown'}</span>
            </div>
          </td>
          <td style="font-size:.75rem;color:var(--t2)">${r.LoanPurpose || 'Other'}</td>
          <td style="font-family:'JetBrains Mono';font-size:.8rem">₹${(+r.LoanAmount || 0).toLocaleString()}</td>
          <td><span class="vbadge ${r.prediction === 1 ? 'hi' : 'lo'}">${r.label || 'N/A'}</span></td>
          <td><button class="pill-btn" onclick="navigate('portfolio')" style="padding:.2rem .5rem;font-size:.6rem">Details</button></td>
        </tr>
      `).join('');
    }
  }

  async function refreshPortfolio() {
    await refreshHistory();
    document.getElementById('port-total').textContent = currentState.stats.total;
    document.getElementById('port-low').textContent = currentState.stats.low_risk;
    document.getElementById('port-high').textContent = currentState.stats.high_risk;

    const tbody = document.getElementById('port-tbody');
    if (tbody && currentState.history && currentState.history.length) {
      tbody.innerHTML = currentState.history.map(r => `
        <tr>
          <td><strong style="font-family:'JetBrains Mono';font-size:.7rem">${r.id || 'N/A'}</strong></td>
          <td><span style="font-weight:600">${r.FullName || 'Unknown'}</span></td>
          <td style="color:var(--t3);font-size:.7rem">${r.timestamp || ''}</td>
          <td>₹${(+r.Income || 0).toLocaleString()}</td>
          <td>₹${(+r.LoanAmount || 0).toLocaleString()}</td>
          <td>${r.CreditScore || 0}</td>
          <td>${r.DTIRatio || 0}</td>
          <td>${r.LoanPurpose || 'Other'}</td>
          <td><span class="vbadge ${r.prediction === 1 ? 'hi' : 'lo'}">${r.label || 'N/A'}</span></td>
        </tr>
      `).join('');
    }
  }

  async function refreshReports() {
    await refreshStats();
    document.getElementById('rep-total').textContent = currentState.stats.total;
    document.getElementById('rep-low').textContent = currentState.stats.low_risk;
    document.getElementById('rep-high').textContent = currentState.stats.high_risk;

    const rate = currentState.stats.default_rate;
    document.getElementById('rep-rate-big').textContent = rate + '%';
    document.getElementById('rep-conf-big').textContent = currentState.stats.avg_confidence + '%';

    // Donut update
    const offset = 220 - (220 * rate / 100);
    const circle = document.getElementById('rep-donut-fg');
    const pctTxt = document.getElementById('rep-donut-pct');
    const bar = document.getElementById('rep-bar');

    if (circle) circle.style.strokeDashoffset = offset;
    if (pctTxt) pctTxt.textContent = rate + '%';
    if (bar) bar.style.width = rate + '%';
  }

  async function refreshModelSpecs() {
    await refreshModelInfo();
    if (!currentState.modelInfo) return;

    const m = currentState.modelInfo;

    // Spec Grid
    document.getElementById('spec-grid').innerHTML = `
      <div class="spec-item"><div class="spec-key">Architecture</div><div class="spec-val">${m.model_type}</div></div>
      <div class="spec-item"><div class="spec-key">Trees</div><div class="spec-val">${m.num_trees}</div></div>
      <div class="spec-item"><div class="spec-key">Learning Rate</div><div class="spec-val">${m.params.learning_rate}</div></div>
      <div class="spec-item"><div class="spec-key">Max Depth</div><div class="spec-val">${m.params.max_depth || 'Auto'}</div></div>
      <div class="spec-item"><div class="spec-key">Leaves</div><div class="spec-val">${m.params.num_leaves}</div></div>
      <div class="spec-item"><div class="spec-key">Regularization</div><div class="spec-val">L1: ${m.params.lambda_l1}, L2: ${m.params.lambda_l2}</div></div>
    `;

    // Encoders
    const encDiv = document.getElementById('encoder-body');
    encDiv.innerHTML = Object.entries(m.categorical_encoders).map(([k, v]) => `
      <div class="enc-row">
        <div class="enc-key">${k}</div>
        <div class="enc-vals">${v.map(val => `<span class="enc-pill">${val}</span>`).join('')}</div>
      </div>
    `).join('');

    // Feature Importance
    const impDiv = document.getElementById('feat-imp-body');
    impDiv.innerHTML = m.feature_importance.map(f => `
      <div class="fi-row">
        <div class="fi-top"><span class="fi-name">${f.feature}</span><span class="fi-pct">${f.pct}%</span></div>
        <div class="fi-track"><div class="fi-bar" style="width:${f.pct}%"></div></div>
      </div>
    `).join('');

    // Feature Tags
    document.getElementById('feat-tags').innerHTML = m.features.map(f => `<span class="ftag">${f}</span>`).join('');
  }

  // ── ANALYSIS ENGINE (PREDICT) ──────────────────────────────────
  if (form) {
    form.addEventListener('submit', async e => {
      e.preventDefault();
      clearErr();

      const data = {};
      new FormData(form).forEach((v, k) => { data[k] = v; });

      setLoad(true);
      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        const r = await res.json();
        if (r.error) { showErr(r.error); return; }
        renderResult(r, data);
      } catch (err) {
        showErr('Analysis failed. Check if model is loaded on server.');
      } finally {
        setLoad(false);
      }
    });
  }

  function renderResult(r, inp) {
    const isHigh = r.prediction === 1;

    rc.className = 'rc on ' + (isHigh ? 'high' : 'low');
    document.getElementById('rc-pill').textContent = isHigh ? 'HIGH RISK' : 'LOW RISK';
    document.getElementById('rc-verdict').textContent = r.label;
    document.getElementById('rc-conf').innerText = r.confidence + '%';

    // Donut
    const pct = r.prob_high_risk;
    const offset = CIRC - (CIRC * pct / 100);
    setTimeout(() => {
      if (dFg) dFg.style.strokeDashoffset = offset;
      document.getElementById('pb-low').style.width = r.prob_low_risk + '%';
      document.getElementById('pb-high').style.width = r.prob_high_risk + '%';
    }, 100);

    animCount(dPct, 0, pct, 900, v => v.toFixed(1) + '%');
    document.getElementById('pv-low').textContent = r.prob_low_risk + '%';
    document.getElementById('pv-high').textContent = r.prob_high_risk + '%';

    idle.classList.add('off');
    sc.classList.add('on');
    buildSignals(inp, isHigh);
    generateAIIntelligence(r, inp);
  }

  function buildSignals(inp, isHigh) {
    const score = +inp.CreditScore;
    const dti = +inp.DTIRatio;
    const inc = +inp.Income || 1;
    const loan = +inp.LoanAmount;
    const ltv = loan / inc;

    const items = [
      { n: 'Credit Score', v: score, g: score >= 700 ? 'good' : score >= 600 ? 'warn' : 'bad', t: score >= 700 ? 'Excellent' : score >= 600 ? 'Fair' : 'Poor' },
      { n: 'DTI Ratio', v: (dti * 1).toFixed(2), g: dti <= .35 ? 'good' : dti <= .5 ? 'warn' : 'bad', t: dti <= .35 ? 'Healthy' : dti <= .5 ? 'Moderate' : 'High' },
      { n: 'LTV Ratio', v: ltv.toFixed(2), g: ltv <= 2 ? 'good' : ltv <= 4 ? 'warn' : 'bad', t: ltv <= 2 ? 'Safe' : ltv <= 4 ? 'Moderate' : 'Aggressive' },
      { n: 'Employment', v: inp.EmploymentType, g: inp.EmploymentType === 'Full-time' ? 'good' : 'warn', t: inp.EmploymentType }
    ];

    document.getElementById('sb').innerHTML = items.map(s => `
      <div class="si">
        <div class="si-n">${s.n}</div>
        <div class="si-v">${s.v}</div>
        <span class="si-b ${s.g}">${s.t}</span>
      </div>`).join('');
  }

  // ── HELPERS ───────────────────────────────────────────────────
  function setLoad(on) {
    btn.classList.toggle('loading', on);
    btn.disabled = on;
  }
  function showErr(msg) {
    errTxt.textContent = msg;
    errBox.classList.add('on');
  }
  function clearErr() { errBox.classList.remove('on'); }

  function animCount(el, from, to, ms, fn = v => v.toFixed(1)) {
    const t0 = performance.now();
    const tick = now => {
      const p = Math.min((now - t0) / ms, 1);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = fn(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function generateAIIntelligence(r, inp) {
    const card = document.getElementById('ai-intel-card');
    const text = document.getElementById('ai-intel-text');
    if (!card || !text) return;

    const name = inp.FullName || 'the applicant';
    const isHigh = r.prediction === 1;

    const insights = {
      high: [
        `Analysis indicates that ${name} presents significant risk factors, primarily driven by debt-to-income metrics and credit history. Portfolio exposure should be minimized.`,
        `Risk modeling identifies critical vulnerabilities in ${name}'s application. Structural debt levels exceed recommended thresholds for this loan type.`,
        `Low confidence in repayment stability for ${name}. Model highlights multiple high-impact risk drivers that suggest a high probability of default.`
      ],
      low: [
        `Risk assessment for ${name} confirms a stable financial profile. Income-to-loan ratios and credit scores align with low-risk benchmarks.`,
        `Intelligence reports suggest ${name} is a high-quality borrower. Strong positive signals in employment duration provide a significant safety buffer.`,
        `Model interpretation for ${name} shows robust financial health. Core risk drivers remain well within optimal safety ranges.`
      ]
    };

    const pool = isHigh ? insights.high : insights.low;
    const selected = pool[Math.floor(Math.random() * pool.length)];

    text.textContent = selected;
    card.style.display = 'block';
  }

  // Init when DOM is actually ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => navigate('dashboard'));
  } else {
    navigate('dashboard');
  }

})();
