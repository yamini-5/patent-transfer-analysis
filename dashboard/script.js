/* ── Utility ── */
function fmt(v, dec=4) { return (v != null && !isNaN(v)) ? Number(v).toFixed(dec) : '--'; }
function fmtPct(v)     { return (v != null && !isNaN(v)) ? (Number(v)*100).toFixed(1)+'%' : '--'; }
function clamp01(v)    { return Math.max(0, Math.min(1, Number(v)||0)); }

document.addEventListener('DOMContentLoaded', () => {

  /* ── Guard ── */
  if (typeof transferData === 'undefined') {
    console.error('transferData not found. Run export_json.py first.');
    return;
  }

  const data     = transferData;         // array of merged result+decision rows
  const kpis     = typeof dashboardKPIs       !== 'undefined' ? dashboardKPIs       : {};
  const imports  = typeof featureImportances  !== 'undefined' ? featureImportances  : [];

  /* ══════════════════════════════════════════════
     1. STAT CARDS
  ══════════════════════════════════════════════ */
  set('sc-total', kpis.total_pairs ?? data.length);
  set('sc-neg',   kpis.neg_transfer_count ?? data.filter(d=>d.negative_transfer).length);
  set('sc-safe',  kpis.safe_pairs ?? '--');
  set('sc-conf',  fmt(kpis.avg_confidence, 2) ?? '--');
  set('sc-shift', fmt(kpis.avg_label_shift, 2) ?? '--');

  function set(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  /* ══════════════════════════════════════════════
     2. INTERACTIVE TOOL — populate selects
  ══════════════════════════════════════════════ */
  const domains = [...new Set(data.map(d => d.source))].sort();
  const srcSel  = document.getElementById('sourceSelect');
  const tgtSel  = document.getElementById('targetSelect');

  domains.forEach(dom => {
    [srcSel, tgtSel].forEach(sel => {
      const opt = document.createElement('option');
      opt.value = opt.textContent = dom;
      sel.appendChild(opt);
    });
  });
  if (domains.length > 1) tgtSel.value = domains[1];

  /* Update demo result card */
  function updateDemo() {
    const src = srcSel.value;
    const tgt = tgtSel.value;
    const row = data.find(d => d.source === src && d.target === tgt);

    set('resPair', src === tgt ? 'Same domain selected' : `${src} → ${tgt}`);

    if (!row || src === tgt) {
      resetResultCard();
      return;
    }

    /* Feature bars */
    const sim     = clamp01(row.similarity      ?? 0);
    const deltaN  = clamp01((row.delta_f1+1)/2  ?? 0);   // normalise -1..1 → 0..1
    const shift   = clamp01(row.label_shift      ?? 0);
    const vocab   = clamp01(row.vocab_overlap    ?? 0);
    const conf    = clamp01(row.avg_confidence   ?? 0);
    const entN    = clamp01((row.entropy??0)/3);           // rough normalise
    const err     = clamp01(row.error_rate       ?? 0);

    fillFeature('fSim',   fmt(row.similarity,2),    sim*100,   'blue',   'fSimBar');
    fillFeature('fDelta', (row.delta_f1>=0?'+':'')+fmt(row.delta_f1,3), deltaN*100, row.delta_f1>=0?'green':'red', 'fDeltaBar');
    fillFeature('fShift', fmt(row.label_shift,3),   shift*100, 'red',    'fShiftBar');
    fillFeature('fVocab', fmt(row.vocab_overlap,3), vocab*100, 'blue',   'fVocabBar');
    fillFeature('fConf',  fmt(row.avg_confidence,3),conf*100,  'green',  'fConfBar');
    fillFeature('fEnt',   fmt(row.entropy,3),        entN*100,  'yellow', 'fEntBar');
    fillFeature('fErr',   fmtPct(row.error_rate),   err*100,   'red',    'fErrBar');

    /* Meta signals */
    set('sigMeta',     row.meta_prediction ?? '--');
    set('sigMetaConf', fmt(row.meta_confidence,2) ?? '--');
    set('sigEns',      fmt(row.ensemble_score,2)  ?? '--');
    set('sigErr',      row.error_level ?? '--');

    /* Decision badge */
    const badge = document.getElementById('decisionBadge');
    const isSafe = (row.final_decision||'').includes('SAFE');
    badge.textContent  = isSafe ? 'SAFE TO TRANSFER' : 'DO NOT TRANSFER';
    badge.className    = 'decision-badge ' + (isSafe ? 'badge-safe' : 'badge-unsafe');

    /* Reason */
    const rb = document.getElementById('reasonBox');
    document.getElementById('reasonText').textContent = row.decision_reason ?? 'N/A';
    rb.className = 'reason-box ' + (isSafe ? 'safe-reason' : 'block-reason');
  }

  function fillFeature(valId, valText, pct, colorClass, barId) {
    set(valId, valText);
    const bar = document.getElementById(barId);
    if (bar) {
      bar.style.width = pct + '%';
      bar.className   = 'feat-bar-fill ' + colorClass;
    }
  }

  function resetResultCard() {
    ['fSim','fDelta','fShift','fVocab','fConf','fEnt','fErr',
     'sigMeta','sigMetaConf','sigEns','sigErr','reasonText'].forEach(id => set(id,'--'));
    ['fSimBar','fDeltaBar','fShiftBar','fVocabBar','fConfBar','fEntBar','fErrBar']
      .forEach(id => { const b=document.getElementById(id); if(b) b.style.width='0%'; });
    const badge = document.getElementById('decisionBadge');
    badge.textContent = '—'; badge.className = 'decision-badge badge-neutral';
    document.getElementById('reasonBox').className = 'reason-box';
  }

  srcSel.addEventListener('change', updateDemo);
  tgtSel.addEventListener('change', updateDemo);
  updateDemo();

  /* ══════════════════════════════════════════════
     3. FEATURE IMPORTANCE CHART
  ══════════════════════════════════════════════ */
  const impChart = document.getElementById('importanceChart');
  if (impChart && imports.length) {
    const maxImp = Math.max(...imports.map(r => r.rf_importance ?? r.importance ?? 0));
    imports.forEach(row => {
      const imp  = row.rf_importance ?? row.importance ?? 0;
      const pct  = maxImp > 0 ? (imp / maxImp * 100).toFixed(1) : '0';
      const coef = row.lr_coef != null ? (row.lr_coef >= 0 ? '+' : '') + Number(row.lr_coef).toFixed(3) : '--';
      const coefColor = (row.lr_coef ?? 0) >= 0 ? '#4ade80' : '#f87171';
      impChart.insertAdjacentHTML('beforeend', `
        <div class="imp-row">
          <div class="imp-label">${row.feature}</div>
          <div class="imp-bar-wrap"><div class="imp-bar" style="width:${pct}%"></div></div>
          <div class="imp-val">${(imp * 100).toFixed(1)}%</div>
          <div class="imp-coef" style="color:${coefColor};font-size:0.75rem;margin-left:6px;">LR:${coef}</div>
        </div>`);
    });
  } else if (impChart) {
    impChart.textContent = 'Run main.py to generate feature importances.';
    impChart.style.color = '#7a8aaa';
    impChart.style.fontSize = '0.85rem';
  }

  /* ══════════════════════════════════════════════
     4. RESULTS TABLE
  ══════════════════════════════════════════════ */
  const tbody = document.getElementById('resultsBody');
  let currentFilter = 'all';

  function renderTable(filter) {
    tbody.innerHTML = '';
    const rows = filter === 'neg'  ? data.filter(d=>d.negative_transfer) :
                 filter === 'safe' ? data.filter(d=>!d.negative_transfer) :
                 data;

    rows.forEach(row => {
      const isSafe    = !(row.negative_transfer);
      const decision  = row.final_decision ?? '--';
      const pillClass = decision.includes('SAFE') ? 'pill-safe' : 'pill-unsafe';
      const pillText  = decision.includes('SAFE') ? 'Safe' : 'Unsafe';
      const deltaClass= (row.delta_f1<0) ? 'cell-neg' : 'cell-pos';
      const deltaStr  = (row.delta_f1>=0?'+':'')+fmt(row.delta_f1,3);

      tbody.insertAdjacentHTML('beforeend', `
        <tr>
          <td><strong>${row.source} &rarr; ${row.target}</strong></td>
          <td>${fmt(row.similarity,4)}</td>
          <td>${fmt(row.f1_baseline,4)}</td>
          <td>${fmt(row.f1_transfer,4)}</td>
          <td class="${deltaClass}">${deltaStr}</td>
          <td>${fmt(row.label_shift,4)}</td>
          <td>${fmt(row.vocab_overlap,4)}</td>
          <td>${fmt(row.avg_confidence,4)}</td>
          <td>${fmt(row.entropy,4)}</td>
          <td>${fmtPct(row.error_rate)}</td>
          <td><span class="badge-pill ${pillClass}">${pillText}</span></td>
        </tr>`);
    });
  }

  renderTable('all');

  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      renderTable(currentFilter);
    });
  });

  /* ══════════════════════════════════════════════
     5. DECISION REPORT CARDS
  ══════════════════════════════════════════════ */
  const grid = document.getElementById('reportsGrid');
  if (grid) {
    data.forEach(row => {
      const isSafe    = (row.final_decision||'').includes('SAFE');
      const cardClass = isSafe ? 'card-safe' : 'card-unsafe';
      const badgeClass= isSafe ? 'badge-safe' : 'badge-unsafe';
      const badgeTxt  = isSafe ? 'SAFE &#10003;' : 'DO NOT TRANSFER &#10007;';

      const feats = [
        ['Similarity',    fmt(row.similarity,4)],
        ['&#916;F1',      (row.delta_f1>=0?'+':'')+fmt(row.delta_f1,4)],
        ['Label Shift',   fmt(row.label_shift,4)],
        ['Vocab Overlap', fmt(row.vocab_overlap,4)],
        ['Avg Conf',      fmt(row.avg_confidence,4)],
        ['Entropy',       fmt(row.entropy,4)],
        ['Error Rate',    fmtPct(row.error_rate)],
      ];

      const featsHTML = feats.map(([k,v])=>
        `<div class="rc-feat-row"><span class="rc-feat-key">${k}</span><span class="rc-feat-val">${v}</span></div>`
      ).join('');

      grid.insertAdjacentHTML('beforeend', `
        <div class="report-card ${cardClass}">
          <div class="rc-header">
            <div class="rc-pair">${row.source} &rarr; ${row.target}</div>
            <span class="decision-badge ${badgeClass}">${badgeTxt}</span>
          </div>
          <div class="rc-feats">${featsHTML}</div>
          <div class="rc-signals">
            Meta: <strong>${row.meta_prediction??'--'}</strong> &nbsp;|&nbsp;
            Conf: <strong>${fmt(row.meta_confidence,2)}</strong> &nbsp;|&nbsp;
            Ensemble: <strong>${fmt(row.ensemble_score,2)}</strong> &nbsp;|&nbsp;
            Error: <strong>${row.error_level??'--'}</strong>
          </div>
          <div class="rc-reason">${row.decision_reason??'N/A'}</div>
        </div>`);
    });
  }

}); // end DOMContentLoaded

/* ══════════════════════════════════════════════
   TAB SWITCHER (global — called from HTML onclick)
══════════════════════════════════════════════ */
function showTab(tabId, btnEl) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  const tab = document.getElementById(tabId);
  if (tab) tab.classList.add('active');
  if (btnEl) btnEl.classList.add('active');
}
