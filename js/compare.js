/* ==========================================================================
   DraftDB — compare.js  (Player Comparison page)
   Renders both player columns from a data object, draws radar / bars /
   hands / crests / flags / pixel avatars inline, wires the intro motion,
   radar tooltips and the Swap control. Depends on window.DraftDB (base.js)
   but degrades without it.
   ========================================================================== */
(function (win, doc) {
  'use strict';

  var DB = win.DraftDB || {};
  var REDUCED = false;
  try { REDUCED = !!DB.reducedMotion || (win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches); } catch (e) {}
  var raf = win.requestAnimationFrame ? win.requestAnimationFrame.bind(win) : function (f) { return setTimeout(f, 16); };
  var now = function () { return (win.performance && performance.now) ? performance.now() : Date.now(); };
  function $(sel, root) { return (root || doc).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || doc).querySelectorAll(sel)); }
  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  /* ---------------------------------------------------------------------
     Data — everything the two columns need lives here so Swap just
     reverses the slot order and re-renders.
     --------------------------------------------------------------------- */
  var AXES = ['Pull-up 3PA /100', 'Pull-up 3P %', 'Drives /100', 'Drive FG %', 'Passes /100', 'Assist %'];

  var PLAYERS = {
    whitaker: {
      id: 'whitaker', name: 'Jalen Whitaker', team: 'Duke', league: 'ACC', country: 'usa', crest: 'duke',
      verdict: { type: 'equal', label: 'EQUAL', value: '77.19' },
      stats: [
        ['DraftDB Rating', '77.90', 'accent'], ['Potential', '77.86', 'yellow'], ['+/- Statistic', '132.46', 'green'],
        ['Level', '9.00', 'accent'], null, ['Current Index', '429', 'accent'], ['Possible Index', '560', 'red']
      ],
      bio: { born: '02.14.06 / 20', nat: 'USA', pos: ['SF', 'PF'], height: '6\'8" (2.03m)', weight: '218 lb (99kg)', injury: '50', injuryTone: 'accent' },
      radar: [58, 62, 45, 70, 88, 52],
      bars: [['First Step', 60], ['Speed', 65], ['Handle', 60], ['Passing', 60]],
      hands: [{ side: 'right', tone: 'red', pct: 100, desc: 'elite' }, { side: 'left', tone: 'cyan', pct: 50, desc: 'notable' }],
      avatar: [
        '............',
        '....aaaa....',
        '...aaaaaa...',
        '...aabbaa...',
        '...bccccb...',
        '...bccccb...',
        '..obcccc....',
        '....cccc....',
        '.....cc.....',
        '...aaaaaa...',
        '.aaaaccaaaa.',
        'aaaaaccaaaaa'
      ]
    },
    vranic: {
      id: 'vranic', name: 'Nikola Vranić', team: 'Belgrade Wolves', league: 'ABA League', country: 'srb', crest: 'wolves',
      verdict: { type: 'under', label: 'UNDERRATED', value: '32.93' },
      stats: [
        ['DraftDB Rating', '76.90', 'accent'], ['Potential', '78.75', 'yellow'], ['+/- Statistic', '127.90', 'green'],
        ['Level', '12.70', 'accent'], null, ['Current Index', '493', 'accent'], ['Possible Index', '554', 'red']
      ],
      bio: { born: '05.28.05 / 21', nat: 'Serbia', pos: ['SF', 'SG'], height: '6\'8" (2.03m)', weight: '227 lb (103kg)', injury: '25', injuryTone: 'green' },
      radar: [72, 58, 66, 60, 74, 80],
      bars: [['First Step', 65], ['Speed', 75], ['Handle', 65], ['Passing', 70]],
      hands: [{ side: 'left', tone: 'cyan', pct: 60, desc: 'above average' }, { side: 'right', tone: 'red', pct: 100, desc: 'elite' }],
      avatar: [
        '...aaaaaa...',
        '..aaaaaaaa..',
        '..aaaaaaaa..',
        '..abbbbbba..',
        '...bccccb...',
        '...bccccb...',
        '....ccccbo..',
        '....cccc....',
        '.....cc.....',
        '..bbbbbbbb..',
        '.bbbbbaabbbb',
        'bbbbbbaabbbb'
      ]
    }
  };
  var ORDER = ['whitaker', 'vranic'];      /* slot 0 = Player A (left), slot 1 = Player B (right) */
  var slots = ORDER.slice();

  var AVATAR_INK = { a: '#3b3f42', b: '#5f6569', c: '#9aa0a4', o: '#a855f7' };

  /* ---------------------------------------------------------------------
     Pixel bitmaps → SVG rects (crisp, no images)
     --------------------------------------------------------------------- */
  function rectsFromBitmap(rows, map, cell) {
    var out = '';
    rows.forEach(function (row, y) {
      for (var x = 0; x < row.length; x++) {
        var ch = row[x]; if (ch === '.' || ch === ' ') continue;
        var fill = map[ch]; if (!fill) continue;
        out += '<rect x="' + (x * cell) + '" y="' + (y * cell) + '" width="' + cell + '" height="' + cell + '" fill="' + fill + '"/>';
      }
    });
    return out;
  }

  /* pixel hand — 14x14 cell grid, palm facing the viewer. right hand: thumb on the left. */
  var HAND_RECTS = [
    [3, 1, 2, 5], [6, 0, 2, 6], [9, 1, 2, 5], [12, 3, 2, 3],   /* index, middle, ring, pinky */
    [3, 6, 11, 5],                                              /* palm */
    [0, 5, 2, 3], [1, 7, 2, 2], [2, 8, 2, 1],                   /* thumb */
    [5, 11, 7, 3]                                               /* wrist */
  ];
  var HAND_SHADE = [[3, 6, 2, 1], [6, 6, 2, 1], [9, 6, 2, 1], [12, 6, 2, 1], [5, 11, 7, 1]];
  function handSVG(side) {
    var W = 14, s = 3;
    var body = '', shade = '';
    HAND_RECTS.forEach(function (r) {
      var x = side === 'left' ? W - r[0] - r[2] : r[0];
      body += '<rect x="' + (x * s) + '" y="' + (r[1] * s) + '" width="' + (r[2] * s) + '" height="' + (r[3] * s) + '"/>';
    });
    HAND_SHADE.forEach(function (r) {
      var x = side === 'left' ? W - r[0] - r[2] : r[0];
      shade += '<rect x="' + (x * s) + '" y="' + (r[1] * s) + '" width="' + (r[2] * s) + '" height="' + (r[3] * s) + '"/>';
    });
    return '<svg viewBox="0 0 42 42" aria-hidden="true" focusable="false" shape-rendering="crispEdges">' +
      '<g fill="currentColor">' + body + '</g><g fill="rgba(0,0,0,.28)">' + shade + '</g></svg>';
  }

  /* pixel arrow — 14x14, points right; rotated for down / up */
  function arrowSVG(dir) {
    var rot = dir === 'down' ? 90 : dir === 'up' ? -90 : 0;
    var r = '<rect x="0" y="5" width="8" height="4"/>';
    for (var i = 0; i <= 6; i++) r += '<rect x="' + (7 + i) + '" y="' + i + '" width="1" height="' + (14 - 2 * i) + '"/>';
    return '<svg class="verdict__arrow" viewBox="0 0 14 14" aria-hidden="true" focusable="false" shape-rendering="crispEdges">' +
      '<g fill="currentColor" transform="rotate(' + rot + ' 7 7)">' + r + '</g></svg>';
  }

  /* team crest — 40x40: hairline square, 5x5 pixel mark with 2 accent cells */
  var CRESTS = {
    duke: ['.###.', '#...#', '#.o.#', '#...#', '.#o#.'],
    wolves: ['#...#', '.#.#.', '..o..', '.#.#.', '#.o.#']
  };
  function crestSVG(key) {
    var rows = CRESTS[key] || CRESTS.duke;
    return '<svg class="crest" viewBox="0 0 40 40" aria-hidden="true" focusable="false" shape-rendering="crispEdges">' +
      '<rect x="0.5" y="0.5" width="39" height="39" fill="#161819" stroke="rgba(255,255,255,.2)"/>' +
      '<g transform="translate(5 5)">' + rectsFromBitmap(rows, { '#': '#e6e6e6', 'o': '#a855f7' }, 6) + '</g></svg>';
  }

  /* country flag — 40x28, crude stripes */
  function flagSVG(code) {
    var s = '<svg class="flag" viewBox="0 0 40 28" aria-hidden="true" focusable="false" shape-rendering="crispEdges">';
    if (code === 'usa') {
      for (var i = 0; i < 7; i++) s += '<rect x="0" y="' + (i * 4) + '" width="40" height="4" fill="' + (i % 2 ? '#e4e4e4' : '#b3202b') + '"/>';
      s += '<rect x="0" y="0" width="18" height="16" fill="#2f4b8a"/>';
      for (var y = 0; y < 3; y++) for (var x = 0; x < 4; x++) s += '<rect x="' + (2 + x * 4) + '" y="' + (2 + y * 4) + '" width="2" height="2" fill="#e4e4e4"/>';
    } else if (code === 'srb') {
      s += '<rect x="0" y="0" width="40" height="10" fill="#b3202b"/><rect x="0" y="10" width="40" height="9" fill="#2f4b8a"/><rect x="0" y="19" width="40" height="9" fill="#e4e4e4"/>';
      s += '<rect x="6" y="8" width="6" height="8" fill="#e4e4e4"/><rect x="8" y="10" width="2" height="4" fill="#b3202b"/>';
    } else {
      s += '<rect width="40" height="28" fill="#333"/>';
    }
    s += '<rect x="0.5" y="0.5" width="39" height="27" fill="none" stroke="rgba(255,255,255,.18)"/></svg>';
    return s;
  }

  /* ---------------------------------------------------------------------
     Radar (SVG 312x312 — R=124 + label margin, drawn 1:1 in a 312px column;
     8 target rings, tick labels along the vertical axis only, like a target)
     --------------------------------------------------------------------- */
  var RD = { size: 312, cx: 156, cy: 156, R: 124, rings: 8 };
  function polar(angle, r) { return { x: RD.cx + Math.cos(angle) * r, y: RD.cy + Math.sin(angle) * r }; }
  function fmt(n) { return (Math.round(n * 100) / 100).toString(); }
  function tickLabel(i) { var v = 100 * i / RD.rings; return (v % 1 ? v.toFixed(1) : String(v)); }

  function radarSVG(values) {
    var n = AXES.length, i, k, s = '';
    var angles = [];
    for (k = 0; k < n; k++) angles.push(-Math.PI / 2 + k * (2 * Math.PI / n));

    s += '<svg viewBox="0 0 ' + RD.size + ' ' + RD.size + '" role="img" aria-label="Radar of six rate metrics">';
    /* rings: outer → inner so alternating fills read like a target */
    for (i = RD.rings; i >= 1; i--) {
      var r = RD.R * i / RD.rings;
      s += '<circle class="radar__ring" cx="' + RD.cx + '" cy="' + RD.cy + '" r="' + fmt(r) + '" fill="' + (i % 2 === 0 ? '#161819' : '#1c1e1f') + '"/>';
    }
    /* spokes */
    for (k = 0; k < n; k++) {
      var p = polar(angles[k], RD.R);
      s += '<line class="radar__spoke" x1="' + RD.cx + '" y1="' + RD.cy + '" x2="' + fmt(p.x) + '" y2="' + fmt(p.y) + '"/>';
    }
    /* tick labels: one clean column of stamps up the vertical (top) axis, set just inside each ring
       and to the right of the spoke so neither the spoke nor the ring cuts through them */
    for (i = 1; i <= RD.rings; i++) {
      var ty = RD.cy - RD.R * i / RD.rings + 4.5;
      var op = (0.55 + 0.45 * i / RD.rings).toFixed(2);
      s += '<text class="radar__tick" style="opacity:' + op + '" x="' + (RD.cx + 3) + '" y="' + fmt(ty) + '" text-anchor="start" dominant-baseline="middle">' + tickLabel(i) + '</text>';
    }
    /* axis labels, tangential around the perimeter, kept upright */
    for (k = 0; k < n; k++) {
      var la = angles[k], ld = la * 180 / Math.PI;
      var lp = polar(la, RD.R + 22);
      var lrot = Math.sin(la) > 0.01 ? ld - 90 : ld + 90;
      s += '<text class="radar__axis" x="' + fmt(lp.x) + '" y="' + fmt(lp.y) + '" text-anchor="middle" dominant-baseline="middle" transform="rotate(' + fmt(lrot) + ' ' + fmt(lp.x) + ' ' + fmt(lp.y) + ')">' + esc(AXES[k]) + '</text>';
    }
    /* data polygon + vertex squares (scales from center on reveal) */
    var pts = values.map(function (v, idx) { return polar(angles[idx], RD.R * v / 100); });
    s += '<g class="radar__data">';
    s += '<polygon class="radar__poly" points="' + pts.map(function (p) { return fmt(p.x) + ',' + fmt(p.y); }).join(' ') + '"/>';
    pts.forEach(function (p, idx) {
      s += '<g class="radar__pt" data-i="' + idx + '" data-x="' + fmt(p.x) + '" data-y="' + fmt(p.y) + '" tabindex="0" role="img" aria-label="' + esc(AXES[idx]) + ': ' + values[idx] + '">' +
        '<rect class="hit" x="' + fmt(p.x - 8) + '" y="' + fmt(p.y - 8) + '" width="16" height="16"/>' +
        '<rect class="dot" x="' + fmt(p.x - 1.5) + '" y="' + fmt(p.y - 1.5) + '" width="3" height="3"/></g>';
    });
    s += '</g></svg>';
    s += '<div class="radar__tip" role="tooltip" aria-hidden="true"></div>';
    return s;
  }

  /* ---------------------------------------------------------------------
     Column renderer
     --------------------------------------------------------------------- */
  function statsHTML(list) {
    var s = '', gap = false;
    list.forEach(function (row) {
      if (!row) { gap = true; return; }          /* null entry = visual gap before the next row */
      s += '<div class="p-stat' + (gap ? ' p-stat--gap' : '') + '"><span class="p-stat__k">' + esc(row[0]) + '</span>' +
        '<span class="p-stat__v t-' + row[2] + '" data-count="' + row[1] + '">' + esc(row[1]) + '</span></div>';
      gap = false;
    });
    return s;
  }

  function bioHTML(b) {
    var pos = b.pos.map(function (p) { return '<span class="pos">' + esc(p) + '</span>'; }).join('');
    return '' +
      '<div class="bio"><span class="bio__k">Born</span><span class="bio__v">' + esc(b.born) + '</span></div>' +
      '<div class="bio"><span class="bio__k">Nationality</span><span class="bio__v">' + esc(b.nat) + '</span></div>' +
      '<div class="bio"><span class="bio__k">Positions</span><span class="bio__v">' + pos + '</span></div>' +
      '<div class="bio"><span class="bio__k">Height</span><span class="bio__v">' + esc(b.height) + '</span></div>' +
      '<div class="bio"><span class="bio__k">Weight</span><span class="bio__v">' + esc(b.weight) + '</span></div>' +
      '<div class="bio"><span class="bio__k">Injury-Score</span><span class="bio__v t-' + b.injuryTone + '" data-count="' + b.injury + '">' + esc(b.injury) + '</span></div>';
  }

  function barsHTML(bars) {
    var s = '<div class="bars">';
    bars.forEach(function (b, i) {
      s += '<div class="bar"><span class="bar__label">' + esc(b[0]) + '</span>' +
        '<span class="bar__track" role="img" aria-label="' + esc(b[0]) + ' ' + b[1] + ' of 100"><span class="bar__fill" style="--w:' + b[1] + '%" data-delay="' + (i * 100) + '">' +
        '<span class="bar__val" data-count="' + b[1] + '">' + b[1] + '</span></span></span></div>';
    });
    s += '<div class="bar-axis" aria-hidden="true"><span></span><span class="bar-axis__scale">' +
      [0, 25, 50, 75, 100].map(function (t) { return '<span class="bar-axis__t" style="left:' + t + '%">' + t + '</span>'; }).join('') +
      '</span></div>';
    s += '</div>';
    return s;
  }

  function handsHTML(hands) {
    var s = '<div class="hands"><span class="hands__legend" aria-hidden="true">Hands</span><div class="hands__grid">';
    hands.forEach(function (h) {
      s += '<div class="hand hand--' + h.tone + '">' +
        '<div class="hand__side">' + h.side + ' hand</div>' + handSVG(h.side) +
        '<div class="hand__pct"><span data-count="' + h.pct + '">' + h.pct + '</span>%</div>' +
        '<div class="hand__desc">' + esc(h.desc) + '</div></div>';
    });
    return s + '</div></div>';
  }

  function columnHTML(p, slot) {
    var letter = slot === 0 ? 'a' : 'b';
    return '' +
      '<div class="p-col p-col--' + letter + '" data-player="' + p.id + '" style="--sx:' + (slot ? '-44px' : '44px') + ';--sy:' + (slot ? '-24px' : '24px') + '">' +
        /* Row 1 — header */
        '<div class="p-row p-head">' +
          '<div class="p-side">' +
            '<div class="p-vtext"><span>' + esc(p.team) + '</span><span>' + esc(p.league) + '</span></div>' +
            '<div class="p-marks">' + crestSVG(p.crest) + flagSVG(p.country) + '</div>' +
          '</div>' +
          '<div class="p-main">' +
            '<h2 class="p-name">' + esc(p.name) + '</h2>' +
            '<div class="p-figure">' +
              '<div class="avatar"><canvas width="96" height="96" aria-label="Pixel avatar of ' + esc(p.name) + '" role="img"></canvas></div>' +
              '<div class="verdict verdict--' + p.verdict.type + '">' + arrowSVG(p.verdict.type === 'under' ? 'down' : p.verdict.type === 'over' ? 'up' : 'right') +
                '<span class="verdict__label">' + esc(p.verdict.label) + '</span>' +
                '<span class="verdict__value" data-count="' + p.verdict.value + '">' + esc(p.verdict.value) + '</span>' +
              '</div>' +
            '</div>' +
          '</div>' +
          '<span class="p-rule" aria-hidden="true"></span>' +
          '<div class="p-stats">' + statsHTML(p.stats) + '</div>' +
        '</div>' +
        /* Row 2 — bio */
        '<div class="p-row p-bio">' + bioHTML(p.bio) + '</div>' +
        /* Row 3 — charts */
        '<div class="p-row p-charts">' +
          '<div class="radar">' + radarSVG(p.radar) + '</div>' +
          '<div class="p-charts__right">' + barsHTML(p.bars) + handsHTML(p.hands) + '</div>' +
        '</div>' +
      '</div>';
  }

  /* ---------------------------------------------------------------------
     Motion helpers
     --------------------------------------------------------------------- */
  var loops = [];                                   /* running rAF loops, cancelled on re-render */
  function stopLoops() { loops.forEach(function (l) { l.stop = true; }); loops = []; }

  function countUp(el, opts) {
    opts = opts || {};
    var target = el.getAttribute('data-count'); if (target == null) return;
    var to = parseFloat(target); if (isNaN(to)) return;
    var dec = (target.split('.')[1] || '').length;
    var duration = opts.duration || 1000, delay = opts.delay || 0;
    if (REDUCED) { el.textContent = to.toFixed(dec); return; }
    el.textContent = (0).toFixed(dec);
    var t0 = null, state = { stop: false }; loops.push(state);
    function step(t) {
      if (state.stop) return;
      if (t0 === null) t0 = t;
      var k = Math.min(1, (t - t0 - delay) / duration);
      if (k < 0) { raf(step); return; }
      el.textContent = (to * easeOut(k)).toFixed(dec);
      if (k < 1) raf(step); else el.textContent = to.toFixed(dec);
    }
    raf(step);
  }

  /* avatar: pixel cells painted one by one (a bright cursor cell leads the draw) */
  function drawAvatar(canvas, rows, step) {
    var dpr = Math.min(2, win.devicePixelRatio || 1);
    var size = 96, cells = 12, cell = size / cells;
    canvas.width = size * dpr; canvas.height = size * dpr;
    var ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    var list = [];
    rows.forEach(function (row, y) {
      for (var x = 0; x < row.length; x++) { var ch = row[x]; if (AVATAR_INK[ch]) list.push([x, y, AVATAR_INK[ch]]); }
    });
    function paint(c) { ctx.fillStyle = c[2]; ctx.fillRect(c[0] * cell, c[1] * cell, cell + 0.5, cell + 0.5); }
    if (REDUCED) { list.forEach(paint); return; }
    var t0 = now(), drawn = 0, state = { stop: false }; loops.push(state);
    (function frame() {
      if (state.stop) return;
      var n = Math.min(list.length, Math.floor((now() - t0) / step) + 1);
      if (n > drawn) {
        for (var i = drawn; i < n; i++) paint(list[i]);
        if (n < list.length) { ctx.fillStyle = 'rgba(255,255,255,.35)'; var last = list[n - 1]; ctx.fillRect(last[0] * cell, last[1] * cell, cell + 0.5, cell + 0.5); }
        drawn = n;
      }
      if (n < list.length) raf(frame); else { ctx.clearRect(0, 0, size, size); list.forEach(paint); }
    })();
  }

  function animateRow(row, k) {
    if (row.__animated) return; row.__animated = true;
    k = k == null ? 1 : k;
    $all('[data-count]', row).forEach(function (el, i) { countUp(el, { duration: 1000 * (k < 1 ? 0.7 : 1), delay: Math.min(i, 8) * 40 * k }); });
    $all('.bar__fill', row).forEach(function (f) {
      var d = (+f.getAttribute('data-delay') || 0) * k;
      f.style.transitionDelay = REDUCED ? '0ms' : d + 'ms';
      var v = $('.bar__val', f); if (v) v.style.transitionDelay = REDUCED ? '0ms' : (d + 380) + 'ms';
      raf(function () { f.classList.add('is-in'); });
    });
    var cv = $('.avatar canvas', row);
    if (cv) { var p = PLAYERS[row.parentNode.getAttribute('data-player')]; if (p) drawAvatar(cv, p.avatar, k < 1 ? 15 : 30); }
  }

  function wireRadar(col) {
    var wrap = $('.radar', col); if (!wrap) return;
    var tip = $('.radar__tip', wrap);
    var p = PLAYERS[col.getAttribute('data-player')];
    function show(pt) {
      var i = +pt.getAttribute('data-i');
      var x = parseFloat(pt.getAttribute('data-x')), y = parseFloat(pt.getAttribute('data-y'));
      tip.innerHTML = esc(AXES[i]).toUpperCase() + '<b>' + p.radar[i] + '</b>';
      tip.style.left = (x / RD.size * 100) + '%';
      tip.style.top = (y / RD.size * 100) + '%';
      tip.classList.add('is-on'); tip.setAttribute('aria-hidden', 'false');
      $all('.radar__pt.is-hot', wrap).forEach(function (o) { o.classList.remove('is-hot'); });
      pt.classList.add('is-hot');
    }
    function hide() { tip.classList.remove('is-on'); tip.setAttribute('aria-hidden', 'true'); $all('.radar__pt.is-hot', wrap).forEach(function (o) { o.classList.remove('is-hot'); }); }
    $all('.radar__pt', wrap).forEach(function (pt) {
      pt.addEventListener('mouseenter', function () { show(pt); });
      pt.addEventListener('mouseleave', hide);
      pt.addEventListener('focus', function () { show(pt); });
      pt.addEventListener('blur', hide);
      pt.addEventListener('touchstart', function () { show(pt); setTimeout(hide, 1600); }, { passive: true });
    });
  }

  /* ---------------------------------------------------------------------
     Render + intro sequence
     --------------------------------------------------------------------- */
  var card, body, divider, selA, selB, swapBtn;
  var revealTimer = null, busy = false;

  function drawDivider(immediate) {
    if (!divider) return;
    if (immediate || REDUCED) { divider.style.transition = 'none'; divider.style.height = '100%'; return; }
    /* collapse WITHOUT a transition (otherwise 100% → 0 itself animates and the draw just reverses it),
       flush layout, then release the transition and grow to 100% */
    divider.style.transition = 'none';
    divider.style.height = '0px';
    void divider.offsetHeight;
    raf(function () { raf(function () { divider.style.transition = ''; divider.style.height = '100%'; }); });
  }

  function render(opts) {
    opts = opts || {};
    var stagger = opts.stagger == null ? 120 : opts.stagger;
    var k = opts.k == null ? 1 : opts.k;
    stopLoops();
    $all('.p-col', body).forEach(function (c) { body.removeChild(c); });
    slots.forEach(function (id, slot) { body.insertAdjacentHTML('beforeend', columnHTML(PLAYERS[id], slot)); });
    var cols = $all('.p-col', body);
    if (opts.pre) cols.forEach(function (c) { c.classList.add('pre'); });
    cols.forEach(wireRadar);
    syncSelects();

    /* reveal rows row-major (A1, B1, A2, B2, A3, B3), 120ms apart */
    var rows = [];
    for (var r = 0; r < 3; r++) cols.forEach(function (c) { var rr = $all('.p-row', c)[r]; if (rr) rows.push(rr); });
    function revealAll() {
      rows.forEach(function (row, i) {
        var go = function () { row.classList.add('is-in'); animateRow(row, k); };
        if (REDUCED) go(); else setTimeout(go, i * stagger);
      });
    }
    if (revealTimer) clearTimeout(revealTimer);
    var started = false;
    var start = function () { if (started) return; started = true; card.classList.add('is-in'); revealAll(); };
    if (opts.immediate) { start(); return cols; }
    card.classList.remove('is-in');
    var begin = function () { if (started) return; drawDivider(false); start(); };
    if (typeof DB.whenVisible === 'function') DB.whenVisible(card, begin, { threshold: 0.05, fallback: 1200 });
    else begin();
    /* belt-and-braces: nothing may stay hidden even if IntersectionObserver never fires */
    revealTimer = setTimeout(function () {
      var late = !started;
      begin();
      rows.forEach(function (row) { row.classList.add('is-in'); animateRow(row, k); });
      if (late) drawDivider(true);
    }, 1500);
    return cols;
  }

  function syncSelects() {
    if (selA) selA.value = slots[0];
    if (selB) selB.value = slots[1];
  }

  function fillSelects() {
    [selA, selB].forEach(function (sel) {
      if (!sel) return;
      sel.innerHTML = ORDER.map(function (id) { return '<option value="' + id + '">' + esc(PLAYERS[id].name) + '</option>'; }).join('');
      sel.addEventListener('change', function () {
        var slot = +sel.getAttribute('data-slot');
        var other = 1 - slot;
        if (sel.value === slots[other]) { swap(); return; }   /* choosing the other column's player = swap */
        slots[slot] = sel.value;
        rerender();
      });
    });
    syncSelects();
  }

  /* two-phase swap: columns slide out across each other, the re-rendered
     columns enter from the opposite side and replay the choreography faster */
  function rerender() {
    if (busy) return; busy = true;
    body.classList.add('is-swapping');
    if (swapBtn) swapBtn.classList.add('is-busy');
    setTimeout(function () {
      var cols = render({ immediate: true, stagger: 60, k: 0.5, pre: !REDUCED });
      body.classList.remove('is-swapping');
      var release = function () { cols.forEach(function (c) { c.classList.add('enter'); c.classList.remove('pre'); }); };
      if (REDUCED) release(); else raf(function () { raf(release); });
      setTimeout(function () {
        release();                                   /* safety: .pre must never linger */
        cols.forEach(function (c) { c.classList.remove('enter'); });
        busy = false;
        if (swapBtn) swapBtn.classList.remove('is-busy');
        syncSelects();
      }, REDUCED ? 30 : 650);
    }, REDUCED ? 0 : 380);
  }

  function swap() { if (busy) return; slots.reverse(); rerender(); }

  function init() {
    card = $('#cmp-card'); body = $('#cmp-body'); if (!card || !body) return;
    divider = $('.cmp-divider', body);
    selA = $('#sel-a'); selB = $('#sel-b'); swapBtn = $('#swap-btn');
    fillSelects();
    if (swapBtn) swapBtn.addEventListener('click', swap);
    render();
  }

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();

  win.DraftDBCompare = { players: PLAYERS, slots: slots, swap: swap, render: render };
})(window, document);
