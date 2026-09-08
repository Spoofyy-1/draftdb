/* ==========================================================================
   DraftDB — runs.js  (Model Runs / backtest page)
   Renders the DATA object below. To update after a new sealed run, edit DATA.
   accuracy = 100 x Spearman rank correlation between a draft order and the
   drafted players' k-season WAR ranking; scouts = the real draft order.
   ========================================================================== */
(function (win, doc) {
  'use strict';

  var DATA = {
    /* headline = expanding window (how the model is used); strict = trained on 2007-2018 only */
    headline: { ai: 48, scouts: 24, won: '6 of 6', strictAi: 42, strictScouts: 26, strictWon: '7 of 7' },
    years: { train: [2007, 2018], test: [2019, 2025], pending: [2026] },
    runs: [
      { when: '2026-09-08 15:30', model: 'EVO gen11 + international-line mask · verified data v4.19 (three seed sets)', tag: 'current', ai: 42, scouts: 26, xai: 48, won: '7 of 7 · expanding 6 of 6' },
      { when: '2026-09-08', model: 'EVO gen11 · verified data v4.4 (audited inputs)', ai: 43, scouts: 26, xai: 45, won: '5 of 7 · expanding 5 of 6' },
      { when: '2026-09-08', model: 'colin branch clean blend · scored on this vault (baseline, kept separate)', ai: 51, scouts: 26, xai: null, won: '7 of 7' },
      { when: '2026-09-06', model: 'EVO gen11 · training window 2007 · data v3.2 (uncertified inputs, superseded)', ai: 45, scouts: 26, xai: 51, won: '5 of 7 · expanding 5 of 6' },
      { when: '2026-09-06', model: 'v2 gen17 genome re-scored on the rebuilt data v3.2', ai: 45, scouts: 26, xai: 50, won: '3 of 7' },
      { when: '2026-09-05 13:11', model: 'BEST v2 gen17 · data v3 (rebuilt international block + measurements) · previously shipped', ai: 44, scouts: 26, xai: null, won: '5 of 7' },
      { when: '2026-09-05 13:08', model: 'BEST v2 gen17 · data v3.0 (rebuilt international block only)', ai: 45, scouts: 26, xai: null, won: '5 of 7' },
      { when: '2026-09-05 04:12', model: 'evolution champion gen86 · data v1 (walk-forward pick, not shipped)', ai: 43, scouts: 26, xai: null, won: '6 of 7' },
      { when: '2026-09-05 02:05', model: 'overnight queue final (hz+covw) · data v1', ai: 44, scouts: 26, xai: null, won: '5 of 7' },
      { when: '2026-09-05 00:51', model: 'BEST v2 gen17 · data v1 (stale international seasons, since corrected)', ai: 46, scouts: 26, xai: null, won: '5 of 7' }
    ],
    /* [class, k seasons scored, drafted players, AI strict %, scouts %, AI expanding % (null = none), training rows expanding] */
    classes: [
      [2019, 5, 58, 52, 40, null, null],
      [2020, 5, 58, 32, 35, 35, 1098],
      [2021, 5, 56, 62, 41, 60, 1210],
      [2022, 4, 52, 50, 26, 57, 1443],
      [2023, 3, 56, 38, 11, 43, 1540],
      [2024, 2, 55, 33, 13, 38, 1641],
      [2025, 1, 57, 37, 18, 38, 1755]
    ]
  };

  function el(tag, cls, txt) { var n = doc.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; }

  function statGrid() {
    var g = doc.getElementById('stat-grid'); if (!g) return;
    [['AI model accuracy', DATA.headline.ai + '%', 'hl', 'expanding window · strict ' + DATA.headline.strictAi + '%'],
     ['NBA scouts accuracy', DATA.headline.scouts + '%', '', 'the real draft order · strict ' + DATA.headline.strictScouts + '%'],
     ['Drafts the AI won', DATA.headline.won, '', 'strict ' + DATA.headline.strictWon]].forEach(function (s) {
      var c = el('div', 'stat'); c.appendChild(el('div', 'stat__k', s[0])); c.appendChild(el('div', 'stat__v' + (s[2] ? ' ' + s[2] : ''), s[1])); c.appendChild(el('div', 'stat__sub', s[3])); g.appendChild(c);
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
    ['When', 'Model', 'AI strict', 'NBA scouts', 'AI expanding', 'Drafts the AI won'].forEach(function (h) { tr.appendChild(el('th', null, h)); });
    th.appendChild(tr); t.appendChild(th);
    var tb = el('tbody');
    DATA.runs.forEach(function (r) {
      var row = el('tr', r.tag ? 'shipped' : null);
      row.appendChild(el('td', 'season', r.when));
      var m = el('td', 'model'); m.textContent = r.model; if (r.tag) { var b = el('span', 'tag', r.tag); m.appendChild(b); } row.appendChild(m);
      row.appendChild(el('td', 'stack', r.ai + '%'));
      row.appendChild(el('td', null, r.scouts + '%'));
      row.appendChild(el('td', 'stack', r.xai == null ? '–' : r.xai + '%'));
      row.appendChild(el('td', 'won', r.won));
      tb.appendChild(row);
    });
    t.appendChild(tb); wrap.appendChild(t); root.appendChild(wrap);
  }

  function classTable() {
    var root = doc.getElementById('class-root'); if (!root) return;
    var wrap = el('div', 'rtable-wrap'), t = el('table', 'rtable'), th = el('thead'), tr = el('tr');
    ['Class', 'Seasons scored', 'Drafted', 'AI strict', 'NBA scouts', 'AI expanding', 'Training rows (expanding)', 'Won'].forEach(function (h) { tr.appendChild(el('th', null, h)); });
    th.appendChild(tr); t.appendChild(th);
    var tb = el('tbody'), sa = 0, ss = 0, sx = 0, nx = 0, won = 0, wonx = 0;
    DATA.classes.forEach(function (c) {
      var w = c[3] > c[4]; won += w ? 1 : 0; sa += c[3]; ss += c[4];
      var wx = c[5] != null && c[5] > c[4]; if (c[5] != null) { sx += c[5]; nx += 1; wonx += wx ? 1 : 0; }
      var row = el('tr', w ? 'goal' : null);
      row.appendChild(el('td', 'season', String(c[0])));
      row.appendChild(el('td', null, String(c[1])));
      row.appendChild(el('td', null, String(c[2])));
      row.appendChild(el('td', 'stack', c[3] + '%'));
      row.appendChild(el('td', null, c[4] + '%'));
      row.appendChild(el('td', 'stack', c[5] == null ? '–' : c[5] + '%'));
      row.appendChild(el('td', null, c[6] == null ? '–' : String(c[6])));
      row.appendChild(el('td', 'edge ' + (w ? 'pos' : 'neg'), w ? 'AI' : 'scouts'));
      tb.appendChild(row);
    });
    var n = DATA.classes.length, sum = el('tr', 'sum');
    [['mean', 'season'], ['', null], ['', null], [Math.round(sa / n) + '%', 'stack'], [Math.round(ss / n) + '%', null], [Math.round(sx / nx) + '%', 'stack'], ['', null], [won + ' of ' + n + ' · expanding ' + wonx + ' of ' + nx, 'edge pos']].forEach(function (x) { sum.appendChild(el('td', x[1], x[0])); });
    tb.appendChild(sum); t.appendChild(tb); wrap.appendChild(t); root.appendChild(wrap);
  }

  function init() { statGrid(); years(); runsTable(); classTable(); }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();
})(window, document);
