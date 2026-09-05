/* ==========================================================================
   DraftDB — runs.js  (Model Runs / STACK ledger page)
   Renders the blind-evaluation ledger from the DATA object below. To update
   after a new sealed run, edit DATA (meta + mutations) — the tables rebuild.
   Columns per row: [season, k, xgb, tabicl, tabfm, STACK, draft, edge]
   ========================================================================== */
(function (win, doc) {
  'use strict';

  var GOAL = 0.20;   /* edge target: stack ≥ draft + 0.20 in a blind season */

  var DATA = {
    meta: {
      updated: '6:16:30 PM',
      ledger: 33,
      goal: 'stack ≥ draft + 0.20 every blind season',
      flags: ['no synthetic', 'selection on walk-forward only', 'vault-locked outcomes']
    },
    cols: ['season', 'k', 'xgb', 'tabicl', 'tabfm', 'STACK', 'draft', 'edge'],
    mutations: [
      {
        id: 'cum5', desc: '5-yr cumulative WAR (baseline)', weights: 'xgb 0.4 / tabicl 0.6 / tabfm 0',
        rows: [
          [2019, 5, 0.432, 0.400, 0.418, 0.420, 0.402, 0.018],
          [2020, 5, 0.287, 0.311, 0.303, 0.318, 0.350, -0.032],
          [2021, 5, 0.614, 0.531, 0.573, 0.582, 0.408, 0.173],
          [2022, 4, 0.475, 0.446, 0.501, 0.471, 0.262, 0.208],
          [2023, 3, 0.194, 0.277, 0.313, 0.275, 0.111, 0.164],
          [2024, 2, 0.542, 0.511, 0.548, 0.551, 0.134, 0.417],
          [2025, 1, 0.323, 0.278, 0.287, 0.310, 0.176, 0.133]
        ],
        blind: [null, null, 0.410, 0.394, 0.420, 0.418, 0.263, 0.154], blindNote: '2/7 seasons ≥ +0.2',
        wf: [null, null, 0.421, 0.436, 0.411, 0.448, 0.282, 0.166]
      },
      {
        id: 'rate', desc: 'WAR per NBA season played', weights: 'xgb 0.3 / tabicl 0.3 / tabfm 0.4',
        rows: [
          [2019, 5, 0.464, 0.424, 0.406, 0.424, 0.431, -0.007],
          [2020, 5, 0.283, 0.305, 0.303, 0.307, 0.349, -0.043],
          [2021, 5, 0.612, 0.552, 0.551, 0.592, 0.435, 0.156],
          [2022, 4, 0.453, 0.443, 0.476, 0.474, 0.261, 0.212],
          [2023, 3, 0.200, 0.294, 0.308, 0.292, 0.123, 0.169],
          [2024, 2, 0.512, 0.476, 0.517, 0.536, 0.135, 0.401],
          [2025, 1, 0.347, 0.254, 0.278, 0.299, 0.176, 0.123]
        ],
        blind: [null, null, 0.410, 0.393, 0.406, 0.418, 0.273, 0.144], blindNote: '2/7 seasons ≥ +0.2',
        wf: [null, null, 0.420, 0.437, 0.419, 0.449, 0.288, 0.162]
      },
      {
        id: 'zseason', desc: 'per-season-index z-scored WAR (development-curve normalized)', weights: 'xgb 0.1 / tabicl 0.9 / tabfm 0',
        rows: [],
        pending: 'Rows not captured in the current snapshot — paste the zseason table and this fills in.'
      }
    ]
  };

  function el(tag, cls, txt) { var n = doc.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; }
  function num(v) { return v == null ? '' : v.toFixed(3); }
  function edgeStr(v) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3); }   /* real minus sign */

  /* one data cell; idx maps to DATA.cols so we can tag season / STACK / edge */
  function cell(v, idx, tag) {
    var td = doc.createElement(tag || 'td');
    var key = DATA.cols[idx];
    if (key === 'season') { td.className = 'season'; td.textContent = v == null ? '' : String(v); return td; }
    if (key === 'k') { td.textContent = v == null ? '' : String(v); return td; }
    if (key === 'STACK') { td.className = 'stack'; td.textContent = num(v); return td; }
    if (key === 'edge') {
      td.className = 'edge ' + (v >= 0 ? 'pos' : 'neg');
      td.textContent = v == null ? '' : edgeStr(v);
      return td;
    }
    td.textContent = num(v);
    return td;
  }

  function summaryRow(vals, label, cls, note) {
    var tr = el('tr', 'sum' + (cls ? ' ' + cls : ''));
    vals.forEach(function (v, i) {
      var td;
      if (i === 0) { td = el('td', 'season', label); }
      else { td = cell(v, i); }
      tr.appendChild(td);
    });
    if (note) { tr.lastChild.appendChild(doc.createTextNode('  · ' + note)); }
    return tr;
  }

  function buildTable(m) {
    var wrap = el('div', 'rtable-wrap');
    var table = el('table', 'rtable');
    // head
    var thead = el('thead'), htr = el('tr');
    DATA.cols.forEach(function (c) { htr.appendChild(el('th', null, c)); });
    thead.appendChild(htr); table.appendChild(thead);
    // body
    var tbody = el('tbody');
    m.rows.forEach(function (r) {
      var tr = el('tr');
      var met = r[7] != null && r[7] >= GOAL;   /* edge column */
      if (met) tr.className = 'goal';
      r.forEach(function (v, i) { tr.appendChild(cell(v, i)); });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    // foot: blind mean + walk-forward mean
    if (m.blind) table.appendChild(summaryRow(m.blind, 'blind mean', null, m.blindNote));
    if (m.wf) table.appendChild(summaryRow(m.wf, 'walk-forward mean (5 seasons)', 'sum--wf'));
    wrap.appendChild(table);
    return wrap;
  }

  function buildMutation(m) {
    var sec = el('section', 'mut');
    var head = el('div', 'mut__head');
    head.appendChild(el('span', 'mut__id', m.id));
    head.appendChild(el('span', 'mut__desc', '— ' + m.desc));
    head.appendChild(el('span', 'mut__w', 'stack weights ' + m.weights));
    sec.appendChild(head);
    if (m.rows && m.rows.length) sec.appendChild(buildTable(m));
    else sec.appendChild(el('div', 'mut__empty', m.pending || 'No rows yet.'));
    return sec;
  }

  function buildMeta() {
    var root = doc.getElementById('runs-meta'); if (!root) return;
    var cells = [
      ['Goal', DATA.meta.goal, true],
      ['Updated', DATA.meta.updated, false],
      ['Sealed blind evaluations', String(DATA.meta.ledger) + ' · ledger', false],
      ['Constraints', DATA.meta.flags.join(' · '), false]
    ];
    cells.forEach(function (c) {
      var cell = el('div', 'runs-meta__cell');
      cell.appendChild(el('div', 'runs-meta__k', c[0]));
      var v = el('div', 'runs-meta__v'); if (c[2]) v.innerHTML = '<span class="hl">' + c[1] + '</span>'; else v.textContent = c[1];
      cell.appendChild(v);
      root.appendChild(cell);
    });
  }

  function init() {
    buildMeta();
    var root = doc.getElementById('runs-root'); if (!root) return;
    DATA.mutations.forEach(function (m) { root.appendChild(buildMutation(m)); });
  }

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();
  win.DraftDBRuns = { data: DATA };
})(window, document);
