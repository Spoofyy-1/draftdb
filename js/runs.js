/* ==========================================================================
   DraftDB — runs.js  (Model Runs / backtest page)
   Renders the DATA object below. To update after a new sealed run, edit DATA.
   accuracy = 100 x Spearman rank correlation between a draft order and the
   drafted players' k-season WAR ranking; scouts = the real draft order.
   ========================================================================== */
(function (win, doc) {
  'use strict';

  var DATA = {
    headline: { ai: 44, scouts: 26, won: '5 of 7' },
    years: { train: [2010, 2018], test: [2019, 2025], pending: [2026] },
    runs: [
      { when: '2026-09-05 13:11', model: 'BEST v2 gen17 · data v3 (rebuilt international block + measurements)', tag: 'shipped', ai: 44, scouts: 26, won: '5 of 7' },
      { when: '2026-09-05 13:08', model: 'BEST v2 gen17 · data v3.0 (rebuilt international block only)', ai: 45, scouts: 26, won: '5 of 7' },
      { when: '2026-09-05 04:12', model: 'evolution champion gen86 · data v1 (walk-forward pick, not shipped)', ai: 43, scouts: 26, won: '6 of 7' },
      { when: '2026-09-05 02:05', model: 'overnight queue final (hz+covw) · data v1', ai: 44, scouts: 26, won: '5 of 7' },
      { when: '2026-09-05 00:51', model: 'BEST v2 gen17 · data v1', ai: 46, scouts: 26, won: '5 of 7' }
    ],
    /* [class, k seasons scored, drafted players, AI %, scouts %, value % NBA teams, value % AI] */
    classes: [
      [2019, 5, 58, 39, 40, 74.6, 73.7],
      [2020, 5, 58, 31, 35, 72.7, 70.4],
      [2021, 5, 56, 63, 41, 79.3, 84.0],
      [2022, 4, 52, 52, 26, 74.7, 81.1],
      [2023, 3, 56, 35, 11, 75.0, 71.6],
      [2024, 2, 55, 60, 13, 55.9, 80.7],
      [2025, 1, 55, 30, 18, 67.0, 75.2]
    ]
  };

  function el(tag, cls, txt) { var n = doc.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; }

  function statGrid() {
    var g = doc.getElementById('stat-grid'); if (!g) return;
    [['AI model accuracy', DATA.headline.ai + '%', 'hl'], ['NBA scouts accuracy', DATA.headline.scouts + '%', ''], ['Drafts the AI won', DATA.headline.won, '']].forEach(function (s) {
      var c = el('div', 'stat'); c.appendChild(el('div', 'stat__k', s[0])); c.appendChild(el('div', 'stat__v' + (s[2] ? ' ' + s[2] : ''), s[1])); g.appendChild(c);
    });
  }

  function years() {
    var y = doc.getElementById('years'); if (!y) return;
    for (var v = DATA.years.train[0]; v <= DATA.years.pending[DATA.years.pending.length - 1]; v++) {
      var cls = v <= DATA.years.train[1] ? 'yr yr--train' : (v <= DATA.years.test[1] ? 'yr yr--test' : 'yr yr--pending');
      var t = v <= DATA.years.train[1] ? 'training class' : (v <= DATA.years.test[1] ? 'test class (blind)' : 'no outcomes yet — board only');
      var c = el('span', cls, String(v)); c.title = t; y.appendChild(c);
    }
    var lg = el('div', 'years__legend'); lg.innerHTML = '<span class="yr yr--train yr--mini">2010</span> training &nbsp; <span class="yr yr--test yr--mini">2019</span> test, scored blind &nbsp; <span class="yr yr--pending yr--mini">2026</span> no outcomes yet'; y.appendChild(lg);
  }

  function runsTable() {
    var root = doc.getElementById('runs-root'); if (!root) return;
    var wrap = el('div', 'rtable-wrap'), t = el('table', 'rtable rtable--runs'), th = el('thead'), tr = el('tr');
    ['When', 'Model', 'AI', 'NBA scouts', 'Drafts the AI won'].forEach(function (h) { tr.appendChild(el('th', null, h)); });
    th.appendChild(tr); t.appendChild(th);
    var tb = el('tbody');
    DATA.runs.forEach(function (r) {
      var row = el('tr', r.tag ? 'shipped' : null);
      row.appendChild(el('td', 'season', r.when));
      var m = el('td', 'model'); m.textContent = r.model; if (r.tag) { var b = el('span', 'tag', r.tag); m.appendChild(b); } row.appendChild(m);
      row.appendChild(el('td', 'stack', r.ai + '%'));
      row.appendChild(el('td', null, r.scouts + '%'));
      row.appendChild(el('td', 'won', r.won));
      tb.appendChild(row);
    });
    t.appendChild(tb); wrap.appendChild(t); root.appendChild(wrap);
  }

  function classTable() {
    var root = doc.getElementById('class-root'); if (!root) return;
    var wrap = el('div', 'rtable-wrap'), t = el('table', 'rtable'), th = el('thead'), tr = el('tr');
    ['Class', 'Seasons scored', 'Drafted', 'AI accuracy', 'NBA scouts', 'Won', 'Value of order · NBA teams', 'Value of order · AI'].forEach(function (h) { tr.appendChild(el('th', null, h)); });
    th.appendChild(tr); t.appendChild(th);
    var tb = el('tbody'), sa = 0, ss = 0, sv1 = 0, sv2 = 0, won = 0;
    DATA.classes.forEach(function (c) {
      var w = c[3] > c[4]; won += w ? 1 : 0; sa += c[3]; ss += c[4]; sv1 += c[5]; sv2 += c[6];
      var row = el('tr', w ? 'goal' : null);
      row.appendChild(el('td', 'season', String(c[0])));
      row.appendChild(el('td', null, String(c[1])));
      row.appendChild(el('td', null, String(c[2])));
      row.appendChild(el('td', 'stack', c[3] + '%'));
      row.appendChild(el('td', null, c[4] + '%'));
      row.appendChild(el('td', 'edge ' + (w ? 'pos' : 'neg'), w ? 'AI' : 'scouts'));
      row.appendChild(el('td', null, c[5].toFixed(1) + '%'));
      row.appendChild(el('td', 'stack', c[6].toFixed(1) + '%'));
      tb.appendChild(row);
    });
    var n = DATA.classes.length, sum = el('tr', 'sum');
    [['mean', 'season'], ['', null], ['', null], [Math.round(sa / n) + '%', 'stack'], [Math.round(ss / n) + '%', null], [won + ' of ' + n, 'edge pos'], [(sv1 / n).toFixed(1) + '%', null], [(sv2 / n).toFixed(1) + '%', 'stack']].forEach(function (x) { sum.appendChild(el('td', x[1], x[0])); });
    tb.appendChild(sum); t.appendChild(tb); wrap.appendChild(t); root.appendChild(wrap);
  }

  function init() { statGrid(); years(); runsTable(); classTable(); }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();
})(window, document);
