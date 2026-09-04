/* ==========================================================================
   DraftDB — home.js  (homepage-only: hit-rate chart, hexagons, benchmark bars,
   visibility-gated counters, frame-driven pixel trees below the fold)
   Relies on window.DraftDB from js/base.js (already initialised on DOMContentLoaded).
   ========================================================================== */
(function (win, doc) {
  'use strict';
  var DB = win.DraftDB || {};
  var REDUCED = !!DB.reducedMotion;
  var SVG = 'http://www.w3.org/2000/svg';

  function el(name, attrs, parent) {
    var n = doc.createElementNS(SVG, name);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function $all(sel, root) { return Array.prototype.slice.call((root || doc).querySelectorAll(sel)); }

  /* ---------------------------------------------------------------------
     whenSeen(target, cb, opts) — fire cb once, the first time `target` is
     actually on screen. base.js's whenVisible carries a 1.5s wall-clock
     fallback (fine for opacity reveals, wrong for one-shot animations: they
     would finish before the reader scrolls to them). This variant uses an
     IntersectionObserver plus a scroll/resize bounds check, so it still works
     when IntersectionObserver is missing or never reports — and fires at
     once when the target is already in view.
     --------------------------------------------------------------------- */
  function inView(node, ratio) {
    var r = node.getBoundingClientRect();
    var vh = win.innerHeight || doc.documentElement.clientHeight;
    if (!r.height) return r.top < vh && r.bottom > 0;
    var visible = Math.min(r.bottom, vh) - Math.max(r.top, 0);
    return visible / Math.min(r.height, vh) >= (ratio || 0.2);
  }
  function whenSeen(target, cb, opts) {
    opts = opts || {};
    var ratio = opts.threshold == null ? 0.2 : opts.threshold;
    var fired = false, io = null, pending = false;
    function fire() {
      if (fired) return; fired = true;
      if (io) { try { io.disconnect(); } catch (e) {} }
      win.removeEventListener('scroll', check); win.removeEventListener('resize', check);
      cb();
    }
    function check() {
      if (fired || pending) return; pending = true;
      win.requestAnimationFrame(function () { pending = false; if (!fired && inView(target, ratio)) fire(); });
    }
    if (inView(target, ratio)) { fire(); return; }
    if ('IntersectionObserver' in win) {
      try {
        io = new IntersectionObserver(function (entries) {
          if (entries.some(function (en) { return en.isIntersecting || en.intersectionRatio > 0; })) fire();
        }, { threshold: Math.min(ratio, 0.5) });
        io.observe(target);
      } catch (e) { io = null; }
    }
    win.addEventListener('scroll', check, { passive: true });
    win.addEventListener('resize', check);
    if (opts.fallback) setTimeout(fire, opts.fallback);
  }

  /* ---------------------------------------------------------------------
     Hit-rate line chart (6.9). Ticks 1,5,10,20,30,60 evenly spaced.
     Curves use monotone cubic interpolation (no overshoot / ripple):
     plateau near the top, then a steady decline like the reference.
     --------------------------------------------------------------------- */
  function buildChart() {
    var svg = doc.getElementById('hitchart');
    if (!svg) return;
    var W = 600, H = 385, L = 62, R = 72, T = 34, B = 66;
    var pw = W - L - R, ph = H - T - B;
    var ticks = [1, 5, 10, 20, 30, 60];
    var series = [
      { key: 'a', name: 'DraftDB',     v: [93.5, 92, 89, 84, 78, 71.4], cls: 'ln--a', lab: 'endl--a', end: '71.4%' },
      { key: 'w', name: 'Consensus',   v: [90, 85, 76, 65, 56, 49],     cls: 'ln--w', lab: 'endl--w', end: '49%' },
      { key: 'y', name: 'Team Boards', v: [86, 78, 66, 53, 44, 38],     cls: 'ln--y', lab: 'endl--y', end: '38%' }
    ];
    function X(i) { return L + (pw * i) / (ticks.length - 1); }
    function Y(v) { return T + ph - (ph * v) / 100; }
    function sign(v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }
    // d3-style monotoneX tangents
    function slope3(x0, y0, x1, y1, x2, y2) {
      var h0 = x1 - x0, h1 = x2 - x1;
      var s0 = (y1 - y0) / (h0 || (h1 < 0 ? -0 : 0)), s1 = (y2 - y1) / (h1 || (h0 < 0 ? -0 : 0));
      var p = (s0 * h1 + s1 * h0) / (h0 + h1);
      return (sign(s0) + sign(s1)) * Math.min(Math.abs(s0), Math.abs(s1), 0.5 * Math.abs(p)) || 0;
    }
    function slope2(x0, y0, x1, y1, t) { var h = x1 - x0; return h ? (3 * (y1 - y0) / h - t) / 2 : t; }
    function smooth(vals) {
      var xs = vals.map(function (_, i) { return X(i); }), ys = vals.map(Y), n = xs.length;
      var d = 'M' + xs[0].toFixed(1) + ' ' + ys[0].toFixed(1);
      var t0, t1;
      for (var i = 0; i < n - 1; i++) {
        t0 = i === 0 ? slope2(xs[0], ys[0], xs[1], ys[1], slope3(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]))
                     : slope3(xs[i - 1], ys[i - 1], xs[i], ys[i], xs[i + 1], ys[i + 1]);
        t1 = i === n - 2 ? slope2(xs[i], ys[i], xs[i + 1], ys[i + 1], t0)
                         : slope3(xs[i], ys[i], xs[i + 1], ys[i + 1], xs[i + 2], ys[i + 2]);
        var dx = (xs[i + 1] - xs[i]) / 3;
        d += ' C' + (xs[i] + dx).toFixed(1) + ' ' + (ys[i] + dx * t0).toFixed(1) + ' ' +
             (xs[i + 1] - dx).toFixed(1) + ' ' + (ys[i + 1] - dx * t1).toFixed(1) + ' ' +
             xs[i + 1].toFixed(1) + ' ' + ys[i + 1].toFixed(1);
      }
      return d;
    }
    // area under DraftDB
    var a = series[0];
    el('path', { 'class': 'area', d: smooth(a.v) + ' L' + X(5).toFixed(1) + ' ' + Y(0).toFixed(1) + ' L' + X(0).toFixed(1) + ' ' + Y(0).toFixed(1) + ' Z' }, svg);
    // axes
    el('line', { 'class': 'ax', x1: L, y1: T, x2: L, y2: T + ph }, svg);
    el('line', { 'class': 'ax', x1: L, y1: T + ph, x2: L + pw + 18, y2: T + ph }, svg);
    // y ticks + labels
    [0, 20, 40, 60, 80, 100].forEach(function (v) {
      var y = Y(v);
      el('line', { 'class': 'tick', x1: L - 4, y1: y, x2: L, y2: y }, svg);
      var t = el('text', { 'class': 'axl', x: L - 8, y: y + 3.5, 'text-anchor': 'end' }, svg);
      t.textContent = String(v);
    });
    var yl = el('text', { 'class': 'axt', x: 16, y: T + ph / 2, 'text-anchor': 'middle', transform: 'rotate(-90 16 ' + (T + ph / 2) + ')' }, svg);
    yl.textContent = 'HIT RATE %';
    // x ticks + labels
    ticks.forEach(function (p, i) {
      var x = X(i);
      el('line', { 'class': 'tick', x1: x, y1: T + ph, x2: x, y2: T + ph + 5 }, svg);
      var t = el('text', { 'class': 'axl', x: x, y: T + ph + 18, 'text-anchor': 'middle' }, svg);
      t.textContent = String(p);
    });
    var xl = el('text', { 'class': 'axt', x: L, y: T + ph + 42, 'text-anchor': 'start' }, svg);
    xl.textContent = 'PICK';
    // lines + diamond end markers + labels
    series.forEach(function (s) {
      var p = el('path', { 'class': 'ln ' + s.cls, d: smooth(s.v) }, svg);
      var len = 900;
      try { len = Math.ceil(p.getTotalLength()) + 2; } catch (e) {}
      p.style.setProperty('--len', String(len));
      var g = el('g', { 'class': 'end' }, svg);
      var ex = X(5), ey = Y(s.v[5]);
      var fill = s.key === 'a' ? '#ff571a' : s.key === 'w' ? '#fff' : '#f9c425';
      el('rect', { x: ex - 3.5, y: ey - 3.5, width: 7, height: 7, fill: fill, transform: 'rotate(45 ' + ex + ' ' + ey + ')' }, g);
      if (s.key === 'a') {
        var l1 = el('text', { 'class': 'endl ' + s.lab, x: ex + 10, y: ey - 6, 'text-anchor': 'start' }, g);
        var l2 = el('text', { 'class': 'endl ' + s.lab, x: ex + 10, y: ey + 6, 'text-anchor': 'start' }, g);
        l1.textContent = 'DraftDB'; l2.textContent = s.end;
      } else {
        var lbl = el('text', { 'class': 'endl ' + s.lab, x: ex + 10, y: ey + 3.5, 'text-anchor': 'start' }, g);
        lbl.textContent = s.end;
      }
    });
    if (REDUCED) { svg.classList.add('is-drawn'); return; }
    whenSeen(svg, function () { svg.classList.add('is-drawn'); }, { threshold: 0.3 });
  }

  /* ---------------------------------------------------------------------
     Concentric hexagons (6.11) — pointy-top, 10 rings, fading outward.
     The outer rings run past the band edges like the reference.
     --------------------------------------------------------------------- */
  function buildHexes() {
    var g = doc.getElementById('hexes');
    if (!g) return;
    var N = 10, R0 = 545, step = 46;
    for (var i = 0; i < N; i++) {
      var r = R0 - i * step, pts = [];
      for (var k = 0; k < 6; k++) {
        var ang = Math.PI / 6 + (Math.PI / 3) * k; // pointy top/bottom => vertical left/right edges
        pts.push((r * Math.cos(ang)).toFixed(1) + ',' + (r * Math.sin(ang)).toFixed(1));
      }
      var op = 0.28 + (i / (N - 1)) * 0.72; // outer dim, inner bright
      el('polygon', { points: pts.join(' '), 'stroke-opacity': op.toFixed(2) }, g);
    }
  }

  /* ---------------------------------------------------------------------
     Benchmark table underline bars (6.13) — fill when the table is seen
     --------------------------------------------------------------------- */
  function buildBench() {
    var table = doc.getElementById('btable');
    if (!table) return;
    var cells = table.querySelectorAll('td[data-v]');
    Array.prototype.forEach.call(cells, function (td) {
      var v = parseFloat(td.getAttribute('data-v'));
      if (isNaN(v)) return;
      var bar = doc.createElement('span');
      bar.className = 'bar';
      bar.style.setProperty('--w', Math.max(0, Math.min(100, v)) + '%');
      td.appendChild(bar);
    });
    if (REDUCED) { table.classList.add('is-in'); return; }
    whenSeen(table, function () { table.classList.add('is-in'); }, { threshold: 0.15 });
  }

  /* ---------------------------------------------------------------------
     Number-roll counters: base.js built the digit columns (trigger=manual);
     re-create them here so the roll plays when the counter is actually seen.
     --------------------------------------------------------------------- */
  function initRolls() {
    if (!DB.rollNumber) return;
    $all('[data-roll]').forEach(function (node) {
      var api = DB.rollNumber(node, { trigger: 'manual' });
      if (!api) return;
      if (REDUCED) { api.play(); return; }
      whenSeen(node, function () { api.play(); }, { threshold: 0.4 });
    });
  }

  /* ---------------------------------------------------------------------
     Use-case accordion: arrow keys move between rows
     --------------------------------------------------------------------- */
  function accKeys() {
    $all('.acc').forEach(function (acc) {
      acc.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
        var heads = $all('.acc__head', acc);
        var i = heads.indexOf(doc.activeElement);
        if (i < 0) return;
        e.preventDefault();
        var n = heads[(i + (e.key === 'ArrowDown' ? 1 : heads.length - 1)) % heads.length];
        if (n) n.focus();
      });
    });
  }

  /* ---------------------------------------------------------------------
     Pixel tree for canvases that start below the fold.
     base.js grows its dendrite on the wall clock from page load, so a tree
     the reader reaches late is already finished (or truncated). This variant
     only counts time while the canvas is on screen: it starts growing the
     moment it scrolls into view and takes ~data-growth ms to fill in, then
     keeps flickering. Same cells / palette / markers as DraftDB.dendrite.
     --------------------------------------------------------------------- */
  var PALETTE = ['#ff571a', '#ff7a3d', '#ffb454', '#f9c425', '#fff1d6'], OLD = '#7a2d10';
  function rand(a, b) { return a + Math.random() * (b - a); }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function pixelTree(canvas) {
    var ds = canvas.dataset || {};
    var o = {
      cell: +ds.cell || 5, gap: 1, origin: ds.origin || 'bottom', x: ds.x ? +ds.x : null,
      trunks: +ds.trunks || 1, trunkWidth: +ds.trunkWidth || 5, density: +ds.density || 1,
      markers: ds.markers != null && ds.markers !== '' ? +ds.markers : 2, growth: +ds.growth || 1200,
      stretch: +ds.stretch || 1 // >1 lets trunks travel further than 60% of the canvas
    };
    var ctx = canvas.getContext('2d'); if (!ctx) return null;
    var pitch = o.cell + o.gap, W = 0, H = 0, cols = 0, rows = 0, reach = 1;
    var cells, map, branches, markers, grown, visible = false, alive = true, lastFlick = 0, raf = 0;
    var seenMs = 0, lastT = 0, steps = 0;

    function size() {
      var r = canvas.getBoundingClientRect();
      var w = Math.round(r.width) || +canvas.getAttribute('width') || 300;
      var h = Math.round(r.height) || +canvas.getAttribute('height') || 200;
      var dpr = Math.min(win.devicePixelRatio || 1, 2);
      W = w; H = h; canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.floor(w / pitch); rows = Math.floor(h / pitch);
    }
    function dir() {
      switch (o.origin) {
        case 'top': return [0, 1]; case 'left': return [1, 0]; case 'right': return [-1, 0];
        case 'bottom-left': return [0.7, -0.7]; case 'bottom-right': return [-0.7, -0.7];
        case 'top-left': return [0.7, 0.7]; case 'top-right': return [-0.7, 0.7];
        default: return [0, -1];
      }
    }
    function origin(i, n) {
      var fx = o.x;
      switch (o.origin) {
        case 'top': return [Math.round(cols * (fx == null ? 0.5 : fx)), 0];
        case 'left': return [0, Math.round(rows * (fx == null ? 0.5 : fx))];
        case 'right': return [cols - 1, Math.round(rows * (fx == null ? 0.5 : fx))];
        case 'bottom-left': return [0, rows - 1]; case 'bottom-right': return [cols - 1, rows - 1];
        case 'top-left': return [0, 0]; case 'top-right': return [cols - 1, 0];
        default: { var x = fx == null ? 0.5 : fx; if (n > 1) x = 0.25 + 0.5 * (i / (n - 1)); return [Math.round(cols * x), rows - 1]; }
      }
    }
    function reset() {
      cells = []; map = {}; branches = []; markers = []; grown = false; seenMs = 0; lastT = 0; steps = 0;
      var d = dir(), a0 = Math.atan2(d[1], d[0]);
      reach = Math.max(1, Math.abs(d[0]) * cols + Math.abs(d[1]) * rows);
      for (var i = 0; i < o.trunks; i++) {
        var p = origin(i, o.trunks);
        branches.push({ x: p[0], y: p[1], a: a0, base: a0, w: o.trunkWidth, gen: 0, life: 0, maxLife: Math.round(reach * rand(0.52, 0.62) * o.stretch), turn: 0 });
      }
    }
    function add(x, y, gen, tip) {
      x = Math.round(x); y = Math.round(y);
      if (x < 0 || y < 0 || x >= cols || y >= rows) return;
      var k = x + ',' + y; if (map[k]) return;
      var r = Math.random();
      var color = r < 0.28 ? PALETTE[0] : r < 0.5 ? PALETTE[1] : r < 0.68 ? PALETTE[2] : r < 0.86 ? PALETTE[3] : PALETTE[4];
      var c = { x: x, y: y, color: color, born: cells.length, alpha: rand(0.7, 1), tip: tip, gen: gen, old: Math.random() < 0.12 };
      map[k] = c; cells.push(c);
    }
    function step() {
      var nb = [];
      for (var i = 0; i < branches.length; i++) {
        var b = branches[i]; b.life++;
        b.turn += rand(-0.22, 0.22) - b.turn * 0.08; b.turn = clamp(b.turn, b.gen === 0 ? -0.3 : -0.55, b.gen === 0 ? 0.3 : 0.55);
        b.a = b.base + b.turn; b.x += Math.cos(b.a); b.y += Math.sin(b.a);
        if (b.x < -1 || b.y < -1 || b.x > cols || b.y > rows) continue;
        var wid = Math.max(1, Math.round(b.w));
        add(b.x, b.y, b.gen, true);
        var px = -Math.sin(b.a), py = Math.cos(b.a);
        for (var k = 1; k < wid; k++) { var side = (k % 2 ? 1 : -1) * Math.ceil(k / 2); add(b.x + px * side, b.y + py * side, b.gen, false); }
        // sparse speckle only — the reference trees are mostly black between thin limbs
        if (Math.random() < 0.03 * o.density) add(b.x + rand(-3, 3), b.y + rand(-2, 2), b.gen + 1, false);
        b.w = Math.max(1, b.w - (b.gen === 0 ? 0.05 : 0.07));
        var mature = b.gen === 0 ? b.life > b.maxLife * 0.25 : b.life > 3;
        var pB = (b.gen === 0 ? 0.17 : 0.11) * o.density * (mature ? 1 : 0);
        if (Math.random() < pB && branches.length + nb.length < 14 && b.gen < 5) {
          b.lastSide = -(b.lastSide || (Math.random() < 0.5 ? -1 : 1));
          var na = b.base + b.lastSide * rand(0.4, 0.9) + b.turn * 0.4;
          nb.push({ x: b.x, y: b.y, a: na, base: na, w: Math.max(1, b.w * 0.35), gen: b.gen + 1, life: 0, maxLife: Math.round(b.maxLife * rand(0.5, 0.9)), turn: 0 });
        }
        if (b.life > b.maxLife) continue;
        nb.push(b);
      }
      branches = nb;
      if (!branches.length) finish();
    }
    function finish() { if (grown) return; grown = true; pickMarkers(true); }   // re-pick on the finished canopy
    /* markers are picked as soon as limbs exist (progressive), on cells clear of the canvas edges */
    function inside(c) { var px = c.x * pitch, py = c.y * pitch; return px >= 16 && px <= W - 56 && py >= 16 && py <= H - 16; }
    function pickMarkers(force) {
      if (!o.markers || cells.length < 40) return;
      var tips = cells.filter(function (c) { return c.tip && c.gen > 0 && inside(c); });
      if (tips.length < 16) {
        if (!force) return;
        if (tips.length < 2) tips = cells.filter(inside);
        if (tips.length < 2) tips = cells.slice();
      }
      markers = [];
      var minGap = 9, n = tips.length;   // first marker from the outer canopy, second lower down, ≥9 cells apart
      for (var i = 0, guard = 0; i < o.markers && guard < 60; guard++) {
        var lo = i === 0 ? Math.floor(n * 0.55) : 0, hi = i === 0 ? n : Math.max(2, Math.floor(n * 0.6));
        var c = tips[lo + ((Math.random() * (hi - lo)) | 0)] || tips[(Math.random() * n) | 0];
        if (!c) break;
        var near = markers.some(function (m) { return Math.abs(m.x - c.x) < minGap && Math.abs(m.y - c.y) < minGap; });
        if (near && guard < 50) continue;
        i++;
        markers.push({ x: c.x, y: c.y, label: (i > 1 ? 'v.' : 'n.') + ((Math.random() * 9000 + 1000) | 0) });
      }
    }
    function drawMarkers() {
      if (!markers.length) return;
      ctx.save(); ctx.lineWidth = 1; ctx.font = '7px JetBrains Mono, ui-monospace, monospace';
      var s = 24;
      var pts = markers.map(function (m) {
        return [clamp(m.x * pitch + o.cell / 2, s / 2 + 1, W - s / 2 - 1), clamp(m.y * pitch + o.cell / 2, s / 2 + 1, H - s / 2 - 1)];
      });
      if (pts.length > 1) {
        ctx.setLineDash([2, 3]); ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.strokeStyle = 'rgba(255,255,255,.8)'; ctx.fillStyle = 'rgba(255,255,255,.85)';
      pts.forEach(function (p, i) {
        var x = Math.round(p[0] - s / 2) + 0.5, y = Math.round(p[1] - s / 2) + 0.5;
        ctx.strokeRect(x, y, s, s);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + s, y + s); ctx.moveTo(x + s, y); ctx.lineTo(x, y + s); ctx.stroke();
        var label = markers[i].label, lw = ctx.measureText(label).width;
        var lx = x + s + 4; if (lx + lw > W - 2) lx = x - 4 - lw;
        ctx.fillText(label, lx, y + s - 2);
      });
      ctx.restore();
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      var n = cells.length;
      for (var i = 0; i < n; i++) {
        var c = cells[i], a = c.alpha;
        if (!grown && n - c.born < 14) a = 1;
        ctx.globalAlpha = a; ctx.fillStyle = c.old ? OLD : c.color;
        ctx.fillRect(c.x * pitch, c.y * pitch, o.cell, o.cell);
      }
      ctx.globalAlpha = 1;
      drawMarkers();
    }
    function flicker() {
      var n = cells.length; if (!n) return;
      var k = Math.max(1, (n * 0.06) | 0);
      for (var i = 0; i < k; i++) { var c = cells[(Math.random() * n) | 0]; c.alpha = rand(0.35, 1); if (Math.random() < 0.05) c.old = !c.old; }
    }
    function loop(t) {
      if (!alive) return;
      raf = win.requestAnimationFrame(loop);
      if (!visible) { lastT = 0; return; }
      var dt = lastT ? Math.min(t - lastT, 100) : 0; lastT = t;
      if (!grown) {
        seenMs += dt;
        var target = Math.floor(seenMs / (o.growth / (reach * 1.1))), guard = 0;
        while (steps < target && guard++ < 12 && !grown) { step(); steps++; }
        if (!markers.length) pickMarkers(false);
        if (!grown && seenMs > o.growth * 1.4 && steps >= target) finish();
        draw();
      } else if (t - lastFlick > 90) { flicker(); lastFlick = t; draw(); }
    }
    function start() {
      size(); reset();
      if (REDUCED) { var g = 0; while (!grown && g++ < 4000) step(); finish(); draw(); return; }
      if (!raf) raf = win.requestAnimationFrame(loop);
    }
    /* resize: re-lay the tree out at the new grid (finished tree regrown synchronously, a growing
       one replayed to the same step) instead of wiping it and regrowing from a stump */
    function relayout() {
      var wasGrown = grown, wasSteps = steps, wasSeen = seenMs, wasMarkers = markers.length;
      size(); reset();
      if (wasGrown || REDUCED) { var g = 0; while (!grown && g++ < 4000) step(); finish(); }
      else {
        for (var i = 0; i < wasSteps && !grown; i++) step();
        steps = wasSteps; seenMs = wasSeen;
        if (wasMarkers) pickMarkers(false);
      }
      draw();
      if (!REDUCED && !raf) raf = win.requestAnimationFrame(loop);
    }
    var rto, io = null;
    function onResize() {
      if (!alive) return;
      clearTimeout(rto);
      rto = setTimeout(function () { if (!alive) return; var r = canvas.getBoundingClientRect(); if (Math.round(r.width) !== W || Math.round(r.height) !== H) relayout(); }, 150);
    }
    win.addEventListener('resize', onResize);
    if ('IntersectionObserver' in win) {
      try { io = new IntersectionObserver(function (en) { visible = en.some(function (e) { return e.isIntersecting; }); }, { threshold: 0.02 }); io.observe(canvas); }
      catch (e) { io = null; visible = true; }
    } else visible = true;
    // belt and braces: if the observer never reports, a bounds check on scroll flips visibility
    var onScroll = function () { if (!visible) { var r = canvas.getBoundingClientRect(); if (r.bottom > 0 && r.top < (win.innerHeight || 0)) visible = true; } };
    win.addEventListener('scroll', onScroll, { passive: true });
    start();
    var api = { canvas: canvas, restart: start, destroy: function () {
      alive = false; clearTimeout(rto);
      win.removeEventListener('resize', onResize); win.removeEventListener('scroll', onScroll);
      if (io) { try { io.disconnect(); } catch (e) {} io = null; }
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
    } };
    canvas.__pixelTree = api;
    return api;
  }

  function lazyTrees() {
    $all('canvas[data-dendrite]').forEach(function (c) {
      var r = c.getBoundingClientRect();
      if (r.top < win.innerHeight * 1.2) return;            // in/near the first viewport: base.js handles it
      if (c.__dendrite && typeof c.__dendrite.destroy === 'function') c.__dendrite.destroy();
      pixelTree(c);
    });
  }

  function init() {
    buildChart();
    buildHexes();
    buildBench();
    initRolls();
    accKeys();
    lazyTrees();
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init);
  else init();
})(window, document);
