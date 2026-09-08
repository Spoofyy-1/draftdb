/* ==========================================================================
   DraftDB — base.js  (shared runtime: smooth scroll, reveal, decode headline,
   number roll, pixel dendrite canvas, marquee, accordion, nav)
   Plain ES2017, no build step. Exposes window.DraftDB.
   ========================================================================== */
(function (win, doc) {
  'use strict';

  var REDUCED = false;
  try { REDUCED = win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}

  var raf = win.requestAnimationFrame ? win.requestAnimationFrame.bind(win) : function (f) { return setTimeout(f, 16); };
  var now = function () { return (win.performance && performance.now) ? performance.now() : Date.now(); };
  function $all(sel, root) { return Array.prototype.slice.call((root || doc).querySelectorAll(sel)); }
  function toEl(x) { return typeof x === 'string' ? doc.querySelector(x) : x; }
  function rand(a, b) { return a + Math.random() * (b - a); }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  /* "when in view" helper with a hard fallback so nothing stays hidden. cb fires exactly once. */
  function whenVisible(el, cb, opts) {
    opts = opts || {};
    var fired = false;
    var fire = function () { if (fired) return; fired = true; cb(); };
    var fallback = setTimeout(fire, opts.fallback == null ? 1500 : opts.fallback);
    if (!('IntersectionObserver' in win)) { fire(); return; }
    try {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting || en.intersectionRatio > 0) { clearTimeout(fallback); io.disconnect(); fire(); }
        });
      }, { threshold: opts.threshold == null ? 0.15 : opts.threshold, rootMargin: opts.rootMargin || '0px 0px -5% 0px' });
      io.observe(el);
    } catch (e) { fire(); }
  }

  var DraftDB = {};
  DraftDB.reducedMotion = REDUCED;
  DraftDB.whenVisible = whenVisible;

  /* Placeholder links (href="#": Log In / Sign Up, plan CTAs, footer legal + social) would otherwise
     run the browser's default "#" navigation and jump the page to the very top. */
  doc.addEventListener('click', function (ev) {
    var a = ev.target && ev.target.closest ? ev.target.closest('a[href]') : null;
    if (a && a.getAttribute('href') === '#') ev.preventDefault();
  });

  /* ---------------------------------------------------------------------
     smoothScroll(opts) — Lenis (if loaded) wired to rAF. Returns instance or null.
     --------------------------------------------------------------------- */
  DraftDB.lenis = null;
  DraftDB.smoothScroll = function (opts) {
    if (REDUCED) return null;
    if (typeof win.Lenis === 'undefined') return null;
    if (DraftDB.lenis) return DraftDB.lenis;
    var lenis;
    try { lenis = new win.Lenis(Object.assign({ lerp: 0.1, smoothWheel: true }, opts || {})); }
    catch (e) { return null; }
    doc.documentElement.classList.add('lenis', 'lenis-smooth');
    function loop(t) { lenis.raf(t); raf(loop); }
    raf(loop);
    // Any link whose hash targets an element on THIS page glides with Lenis — "#pricing" as well as
    // "index.html#pricing" (nav / announcement / footer use the page-qualified form).
    function samePath(p) {
      var norm = function (s) { return (s || '/').replace(/\/index\.html$/, '/'); };
      return norm(p) === norm(location.pathname);
    }
    doc.addEventListener('click', function (ev) {
      var a = ev.target && ev.target.closest ? ev.target.closest('a[href*="#"]') : null;
      if (!a) return;
      if (a.target && a.target !== '_self') return;
      if ((a.host || '') !== (location.host || '')) return;
      if (a.pathname && !samePath(a.pathname)) return;
      var id = a.hash;
      if (!id || id === '#') return;
      var target = null;
      try { target = doc.getElementById(decodeURIComponent(id.slice(1))); } catch (e) { target = null; }
      if (!target) return;
      ev.preventDefault();
      lenis.scrollTo(target, { offset: -70 });
    });
    DraftDB.lenis = lenis;
    return lenis;
  };

  /* ---------------------------------------------------------------------
     reveal(root) — .reveal / [data-reveal] / .reveal-stagger → .is-in
     IntersectionObserver (15% visible, once) + 1.5s fallback that reveals all.
     --------------------------------------------------------------------- */
  DraftDB.reveal = function (root) {
    root = toEl(root) || doc;
    var els = $all('.reveal, [data-reveal], .reveal-stagger', root);
    if (!els.length) return;
    if (REDUCED || !('IntersectionObserver' in win)) { els.forEach(function (el) { el.classList.add('is-in'); }); return; }
    var io;
    try {
      io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting || en.intersectionRatio >= 0.15) { en.target.classList.add('is-in'); io.unobserve(en.target); }
        });
      }, { threshold: 0.15, rootMargin: '0px 0px -5% 0px' });
      els.forEach(function (el) { io.observe(el); });
    } catch (e) { els.forEach(function (el) { el.classList.add('is-in'); }); return; }
    setTimeout(function () {
      els.forEach(function (el) { if (!el.classList.contains('is-in')) el.classList.add('is-in'); });
    }, 1500);
    raf(function () {
      els.forEach(function (el) {
        var r = el.getBoundingClientRect();
        if (r.top < win.innerHeight && r.bottom > 0) el.classList.add('is-in');
      });
    });
  };

  /* ---------------------------------------------------------------------
     decodeText(el, opts) — hero scramble/decode.
     opts: {duration:1100, stagger:80, charset, delay:0, onDone}
     --------------------------------------------------------------------- */
  DraftDB.decodeText = function (el, opts) {
    el = toEl(el); if (!el) return;
    opts = opts || {};
    var duration = opts.duration || 1100, stagger = opts.stagger == null ? 80 : opts.stagger, delay = opts.delay || 0;
    var charset = opts.charset || '01#%&+=-/\\|<>*:;';
    var text = (el.getAttribute('data-decode-text') || el.textContent).replace(/\s+/g, ' ').trim();
    var words = text.split(' ');
    el.classList.add('decode');
    el.setAttribute('aria-label', text);
    el.innerHTML = '';
    var spans = words.map(function (w, i) {
      var word = doc.createElement('span'); word.className = 'decode__word'; word.setAttribute('aria-hidden', 'true');
      var done = doc.createElement('span'); done.className = 'decode__done';
      var scr = doc.createElement('span'); scr.className = 'decode__scr';
      word.appendChild(done); word.appendChild(scr);
      el.appendChild(word);
      if (i < words.length - 1) el.appendChild(doc.createTextNode(' '));
      return { word: w, done: done, scr: scr, start: delay + i * stagger };
    });
    function scramble(n) { var s = ''; for (var i = 0; i < n; i++) s += charset[(Math.random() * charset.length) | 0]; return s; }
    function finish() {
      spans.forEach(function (s) { s.done.textContent = s.word; s.scr.textContent = ''; });
      el.classList.add('is-decoded');
      if (opts.onDone) opts.onDone(el);
    }
    if (REDUCED) { finish(); return; }
    spans.forEach(function (s) { s.done.textContent = ''; s.scr.textContent = scramble(s.word.length); });
    var t0 = now(), lastTick = 0;
    function frame() {
      var t = now() - t0, allDone = true, tick = ((t / 40) | 0) !== lastTick; lastTick = (t / 40) | 0;
      spans.forEach(function (s) {
        var p = clamp((t - s.start) / duration, 0, 1);
        if (p < 1) allDone = false;
        var n = p >= 1 ? s.word.length : Math.floor(easeOut(p) * s.word.length);
        s.done.textContent = s.word.slice(0, n);
        if (tick || p >= 1) s.scr.textContent = p >= 1 ? '' : scramble(s.word.length - n);
      });
      if (allDone) finish(); else raf(frame);
    }
    raf(frame);
  };

  /* ---------------------------------------------------------------------
     rollNumber(el, opts) — NumberFlow-style digit columns.
     Reads el.textContent (or data-roll="71.4%"). Digits become 0-9 columns that
     translate to the target digit; non-digits (. % > K x) stay static.
     opts: {duration:1400, stagger:60, trigger:'visible'|'now'|'manual', from:'0'|'random'}
     Returns {play(), reset()}
     --------------------------------------------------------------------- */
  DraftDB.rollNumber = function (el, opts) {
    el = toEl(el); if (!el) return null;
    opts = opts || {};
    var text = (el.getAttribute('data-roll') || el.textContent).trim();
    if (!el.getAttribute('data-roll')) el.setAttribute('data-roll', text);
    var duration = opts.duration || 1400, stagger = opts.stagger == null ? 60 : opts.stagger;
    el.classList.add('roll');
    el.setAttribute('aria-label', text);
    el.innerHTML = '';
    var cols = [];
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (/[0-9]/.test(ch)) {
        var d = doc.createElement('span'); d.className = 'roll__digit'; d.setAttribute('aria-hidden', 'true');
        var col = doc.createElement('span'); col.className = 'roll__col';
        for (var k = 0; k <= 9; k++) { var s = doc.createElement('span'); s.textContent = String(k); col.appendChild(s); }
        d.appendChild(col); el.appendChild(d);
        cols.push({ col: col, target: +ch, idx: cols.length });
      } else {
        var st = doc.createElement('span'); st.className = 'roll__static'; st.textContent = ch; st.setAttribute('aria-hidden', 'true');
        el.appendChild(st);
      }
    }
    function setTo(c, v, animate) {
      c.col.style.transition = animate ? ('transform ' + duration + 'ms cubic-bezier(.22,1,.36,1) ' + (c.idx * stagger) + 'ms') : 'none';
      c.col.style.transform = 'translate3d(0,' + (-v) + 'em,0)';
    }
    function resetCols() { cols.forEach(function (c) { setTo(c, opts.from === 'random' ? (Math.random() * 10) | 0 : 0, false); }); }
    var played = false;
    function play() {
      if (played) return; played = true;
      if (REDUCED) { cols.forEach(function (c) { setTo(c, c.target, false); }); el.classList.add('is-rolled'); return; }
      void el.offsetWidth;
      raf(function () { cols.forEach(function (c) { setTo(c, c.target, true); }); el.classList.add('is-rolled'); });
    }
    resetCols();
    var trigger = opts.trigger || 'visible';
    if (trigger === 'now') play();
    else if (trigger === 'visible') whenVisible(el, play, { threshold: 0.2 });
    return { play: play, reset: function () { played = false; resetCols(); } };
  };

  /* ---------------------------------------------------------------------
     dendrite(canvas, opts) — pixel branching tree (square cells, growth then flicker).
     opts: {cell:5, gap:1, growth:6000, origin:'bottom'|'top'|'left'|'right'|'bottom-left'|'bottom-right'|'top-left'|'top-right',
            x:0..1 (origin position along the edge), trunks:1, trunkWidth:4, density:1, markers:2,
            palette:[...], old:'#7a2d10', flicker:true}
     data-* equivalents: data-origin, data-x, data-trunks, data-growth, data-cell, data-density, data-markers, data-stretch
     (stretch > 1 lets trunks travel further than ~60% of the canvas, e.g. the footer corner trees)
     Returns {canvas, restart(), destroy()}
     --------------------------------------------------------------------- */
  var PALETTE = ['#a855f7', '#b975fa', '#cf9dfc', '#e6ccfe', '#f7edff'];
  var OLD = '#3b1466';
  DraftDB.dendrite = function (canvas, opts) {
    canvas = toEl(canvas); if (!canvas) return null;
    opts = Object.assign({
      cell: 5, gap: 1, growth: 6000, origin: 'bottom', trunks: 1, trunkWidth: 5, density: 1, markers: 2,
      palette: PALETTE, old: OLD, flicker: true, x: null, stretch: 1
    }, opts || {});
    (function readData() {
      var ds = canvas.dataset || {};
      if (ds.origin) opts.origin = ds.origin;
      if (ds.markers != null && ds.markers !== '') opts.markers = +ds.markers;
      if (ds.trunks) opts.trunks = +ds.trunks;
      if (ds.growth) opts.growth = +ds.growth;
      if (ds.cell) opts.cell = +ds.cell;
      if (ds.x) opts.x = +ds.x;
      if (ds.density) opts.density = +ds.density;
      if (ds.trunkWidth) opts.trunkWidth = +ds.trunkWidth;
      if (ds.stretch) opts.stretch = +ds.stretch;
    })();
    var ctx = canvas.getContext('2d');
    if (!ctx) return null;
    var pitch = opts.cell + opts.gap;
    var W = 0, H = 0, cols = 0, rows = 0, dpr = 1;
    var cells = [], cellMap = {}, branches = [], markers = [], grown = false, lastTick = 0, lastFlick = 0, rafId = 0, alive = true, visible = true, elapsed = 0, lastT = 0;

    function size() {
      var r = canvas.getBoundingClientRect();
      var w = Math.round(r.width) || +canvas.getAttribute('width') || 300;
      var h = Math.round(r.height) || +canvas.getAttribute('height') || 200;
      dpr = Math.min(win.devicePixelRatio || 1, 2);
      W = w; H = h;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.floor(w / pitch); rows = Math.floor(h / pitch);
    }

    function baseDir() {
      switch (opts.origin) {
        case 'top': return [0, 1];
        case 'left': return [1, 0];
        case 'right': return [-1, 0];
        case 'bottom-left': return [0.7, -0.7];
        case 'bottom-right': return [-0.7, -0.7];
        case 'top-left': return [0.7, 0.7];
        case 'top-right': return [-0.7, 0.7];
        default: return [0, -1];
      }
    }
    function baseAngle() { var d = baseDir(); return Math.atan2(d[1], d[0]); }
    function originPoint(i, n) {
      var fx = opts.x;
      switch (opts.origin) {
        case 'top': return [Math.round(cols * (fx == null ? 0.5 : fx)), 0];
        case 'left': return [0, Math.round(rows * (fx == null ? 0.5 : fx))];
        case 'right': return [cols - 1, Math.round(rows * (fx == null ? 0.5 : fx))];
        case 'bottom-left': return [0, rows - 1];
        case 'bottom-right': return [cols - 1, rows - 1];
        case 'top-left': return [0, 0];
        case 'top-right': return [cols - 1, 0];
        default: {
          var x = fx == null ? 0.5 : fx;
          if (n > 1) x = 0.25 + 0.5 * (i / (n - 1));
          return [Math.round(cols * x), rows - 1];
        }
      }
    }

    function reset() {
      cells = []; cellMap = {}; branches = []; markers = []; grown = false;
      var a0 = baseAngle(), d = baseDir();
      // reach = how many cells the trunk could travel along its heading before leaving the canvas
      var reach = Math.abs(d[0]) * cols + Math.abs(d[1]) * rows;
      for (var i = 0; i < opts.trunks; i++) {
        var o = originPoint(i, opts.trunks);
        branches.push({ x: o[0], y: o[1], a: a0, base: a0, w: opts.trunkWidth, gen: 0, life: 0,
          maxLife: Math.round(reach * rand(0.52, 0.62) * (opts.stretch || 1)), turn: 0 });
      }
      elapsed = 0; lastT = 0; lastTick = 0;
    }

    function addCell(x, y, gen, tip) {
      x = Math.round(x); y = Math.round(y);
      if (x < 0 || y < 0 || x >= cols || y >= rows) return;
      var key = x + ',' + y;
      if (cellMap[key]) return;
      var r = Math.random();
      var color = r < 0.28 ? opts.palette[0] : r < 0.5 ? opts.palette[1] : r < 0.68 ? opts.palette[2] : r < 0.86 ? opts.palette[3] : opts.palette[4];
      var c = { x: x, y: y, color: color, born: cells.length, alpha: rand(0.7, 1), tip: tip, gen: gen, old: Math.random() < 0.12 };
      cellMap[key] = c; cells.push(c);
    }

    function step() {
      var nb = [];
      for (var i = 0; i < branches.length; i++) {
        var b = branches[i];
        b.life++;
        b.turn += rand(-0.22, 0.22) - b.turn * 0.08; b.turn = clamp(b.turn, b.gen === 0 ? -0.3 : -0.55, b.gen === 0 ? 0.3 : 0.55);
        b.a = b.base + b.turn;
        b.x += Math.cos(b.a); b.y += Math.sin(b.a);
        if (b.x < -1 || b.y < -1 || b.x > cols || b.y > rows) continue;
        var wid = Math.max(1, Math.round(b.w));
        addCell(b.x, b.y, b.gen, true);
        var px = -Math.sin(b.a), py = Math.cos(b.a);
        for (var k = 1; k < wid; k++) {
          var side = (k % 2 ? 1 : -1) * Math.ceil(k / 2);
          addCell(b.x + px * side, b.y + py * side, b.gen, false);
        }
        // sparse speckle only (the reference trees are lightning-like, mostly black between limbs)
        if (Math.random() < 0.03 * opts.density) addCell(b.x + rand(-3, 3), b.y + rand(-2, 2), b.gen + 1, false);
        b.w = Math.max(1, b.w - (b.gen === 0 ? 0.05 : 0.07));
        // trunk only branches once it has cleared ~30% of its run; side branches after a few cells
        var mature = b.gen === 0 ? b.life > b.maxLife * 0.25 : b.life > 3;
        var pBranch = (b.gen === 0 ? 0.17 : 0.11) * opts.density * (mature ? 1 : 0);
        if (Math.random() < pBranch && branches.length + nb.length < 14 && b.gen < 5) {
          b.lastSide = -(b.lastSide || (Math.random() < 0.5 ? -1 : 1));   // alternate sides
          var na = b.base + b.lastSide * rand(0.4, 0.9) + b.turn * 0.4;
          // thin limbs that travel far: child width 35% of the parent, life up to 90% of the parent's
          nb.push({ x: b.x, y: b.y, a: na, base: na, w: Math.max(1, b.w * 0.35), gen: b.gen + 1, life: 0,
            maxLife: Math.round(b.maxLife * rand(0.5, 0.9)), turn: 0 });
        }
        if (b.life > b.maxLife) continue;
        nb.push(b);
      }
      branches = nb;
      if (!branches.length) finish();
    }

    function finish() {
      if (grown) return;
      grown = true;
      pickMarkers(true);      // re-pick on the finished canopy (the early picks sit near the trunk)
    }

    /* inspector markers: picked as soon as the trunk has limbs (so a static capture always shows
       them), kept clear of the canvas edges so the 24px box + label never clip. */
    function inside(c) {
      var px = c.x * pitch, py = c.y * pitch;
      return px >= 16 && px <= W - 56 && py >= 16 && py <= H - 16;
    }
    function pickMarkers(force) {
      if (!opts.markers || cells.length < 40) return;
      var tips = cells.filter(function (c) { return c.tip && c.gen > 0 && inside(c); });
      if (tips.length < 16) {                                 // wait until the limbs have spread out
        if (!force) return;                                   // not ready yet — try again next frame
        if (tips.length < 2) tips = cells.filter(inside);
        if (tips.length < 2) tips = cells.slice();
      }
      markers = [];
      // tips are in birth order, so the tail of the list is the outer canopy: first marker high on a
      // limb, second lower down (like the reference), never closer than ~9 cells to each other
      var minGap = 9, n = tips.length;
      for (var i = 0, guard = 0; i < opts.markers && guard < 60; guard++) {
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
      ctx.save();
      ctx.lineWidth = 1;
      ctx.font = '7px JetBrains Mono, ui-monospace, monospace';
      var s = 24;
      var pts = markers.map(function (m) {
        return [clamp(m.x * pitch + opts.cell / 2, s / 2 + 1, W - s / 2 - 1), clamp(m.y * pitch + opts.cell / 2, s / 2 + 1, H - s / 2 - 1)];
      });
      if (pts.length > 1) {
        ctx.setLineDash([2, 3]); ctx.strokeStyle = 'rgba(255,255,255,.5)';
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.strokeStyle = 'rgba(255,255,255,.8)'; ctx.fillStyle = 'rgba(255,255,255,.85)';
      pts.forEach(function (p, i) {
        var x = Math.round(p[0] - s / 2) + 0.5, y = Math.round(p[1] - s / 2) + 0.5;
        ctx.strokeRect(x, y, s, s);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + s, y + s); ctx.moveTo(x + s, y); ctx.lineTo(x, y + s); ctx.stroke();
        var label = markers[i].label, lw = ctx.measureText(label).width;
        var lx = x + s + 4; if (lx + lw > W - 2) lx = x - 4 - lw;   // flip the label left near the right edge
        ctx.fillText(label, lx, y + s - 2);
      });
      ctx.restore();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      var n = cells.length;
      for (var i = 0; i < n; i++) {
        var c = cells[i];
        var a = c.alpha;
        if (!grown) { var age = n - c.born; if (age < 14) a = 1; }
        ctx.globalAlpha = a;
        ctx.fillStyle = c.old ? opts.old : c.color;
        ctx.fillRect(c.x * pitch, c.y * pitch, opts.cell, opts.cell);
      }
      ctx.globalAlpha = 1;
      drawMarkers();
    }

    function flicker() {
      if (!opts.flicker || REDUCED) return;
      var n = cells.length; if (!n) return;
      var k = Math.max(1, (n * 0.06) | 0);
      for (var i = 0; i < k; i++) {
        var c = cells[(Math.random() * n) | 0];
        c.alpha = rand(0.35, 1);
        if (Math.random() < 0.05) c.old = !c.old;
      }
    }

    function loop() {
      if (!alive) return;
      rafId = raf(loop);
      var t = now();
      if (!visible) { lastT = 0; return; }                 // offscreen: the growth clock pauses (no catch-up truncation)
      var dt = lastT ? Math.min(t - lastT, 100) : 0; lastT = t;
      if (!grown) {
        elapsed += dt;                                     // only time spent on screen counts toward growth
        var d = baseDir(), reach = Math.abs(d[0]) * cols + Math.abs(d[1]) * rows;
        var stepsTarget = Math.floor(elapsed / (opts.growth / (reach * 1.1)));
        var guard = 0;
        while (lastTick < stepsTarget && guard++ < 12 && !grown) { step(); lastTick++; }
        if (!markers.length) pickMarkers(false);          // progressive: markers appear once limbs exist
        // hard stop only once no catch-up steps are pending, so a return visit never strands a stump
        if (!grown && elapsed > opts.growth * 1.4 && lastTick >= stepsTarget) finish();
        draw();
      } else if (t - lastFlick > 90) { flicker(); lastFlick = t; draw(); }
    }

    function start() {
      size(); reset();
      if (REDUCED) {
        var guard = 0; while (!grown && guard++ < 4000) step();
        finish(); draw();
        return;
      }
      if (!rafId) loop();
    }

    /* A size change re-lays the tree out at the new grid instead of wiping it and regrowing from a
       stump: a finished tree is regrown synchronously, one still growing is replayed to the same step
       and carries on from there. */
    function relayout() {
      var wasGrown = grown, wasSteps = lastTick, wasElapsed = elapsed, wasMarkers = markers.length;
      size(); reset();
      if (wasGrown || REDUCED) {
        var guard = 0; while (!grown && guard++ < 4000) step();
        finish();
      } else {
        for (var i = 0; i < wasSteps && !grown; i++) step();
        lastTick = wasSteps; elapsed = wasElapsed;
        if (wasMarkers) pickMarkers(false);
      }
      draw();
      if (!REDUCED && !rafId) loop();
    }
    var rto;
    function onResize() {
      if (!alive) return;
      clearTimeout(rto);
      rto = setTimeout(function () {
        if (!alive) return;
        var r = canvas.getBoundingClientRect();
        if (Math.round(r.width) !== W || Math.round(r.height) !== H) relayout();
      }, 150);
    }
    var ro = null, io = null;
    win.addEventListener('resize', onResize);
    if ('ResizeObserver' in win) { try { ro = new ResizeObserver(onResize); ro.observe(canvas); } catch (e) { ro = null; } }
    if ('IntersectionObserver' in win) {
      try { io = new IntersectionObserver(function (en) { visible = en.some(function (e) { return e.isIntersecting; }); }, { rootMargin: '100px' }); io.observe(canvas); } catch (e) { io = null; }
    }

    start();
    var api = {
      canvas: canvas,
      restart: start,
      stats: function () { return { cells: cells.length, branches: branches.length, grown: grown, elapsed: Math.round(elapsed), steps: lastTick, cols: cols, rows: rows, visible: visible, w: W, h: H }; },
      destroy: function () {
        alive = false; clearTimeout(rto);
        win.removeEventListener('resize', onResize);
        if (ro) { try { ro.disconnect(); } catch (e) {} ro = null; }
        if (io) { try { io.disconnect(); } catch (e) {} io = null; }
        if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
        if (canvas.__dendrite === api) canvas.__dendrite = null;
      }
    };
    canvas.__dendrite = api;
    return api;
  };

  /* ---------------------------------------------------------------------
     sprout(canvas, opts) — pixel "draft stock": a rising arrow that grows from the
     bottom-left, with inspector squares pinning a climbing WAR projection along the
     way (a prospect trending up every step). Square cells, growth over on-screen time,
     then a gentle flicker. Replaces the earlier dendrite tree; same [data-dendrite]
     canvases, same visibility-gated growth clock.
     data-* : data-growth (ms), data-cell, data-accent (hex -> tinted team ramp).
     --------------------------------------------------------------------- */
  function _mix(hex, to, t) {
    function parse(h){ h = String(h).replace('#',''); if (h.length===3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2]; return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)]; }
    var a = parse(hex), b = parse(to);
    return '#' + [0,1,2].map(function(i){ return ('0'+Math.round(a[i] + (b[i]-a[i])*t).toString(16)).slice(-2); }).join('');
  }
  function rampFromHex(hex) { return [hex, _mix(hex,'#ffffff',0.28), _mix(hex,'#ffffff',0.52), _mix(hex,'#ffffff',0.72), _mix(hex,'#ffffff',0.9)]; }
  DraftDB.sprout = function (canvas, opts) {
    canvas = toEl(canvas); if (!canvas) return null;
    opts = Object.assign({ cell: 5, gap: 1, growth: 2600, palette: PALETTE, old: OLD, flicker: true }, opts || {});
    var ds = canvas.dataset || {};
    if (ds.growth) opts.growth = +ds.growth;
    if (ds.cell) opts.cell = +ds.cell;
    if (ds.accent) { opts.palette = rampFromHex(ds.accent); opts.old = _mix(ds.accent, '#000000', 0.62); }
    var ctx = canvas.getContext('2d'); if (!ctx) return null;
    var pitch = opts.cell + opts.gap;
    var W = 0, H = 0, cols = 0, rows = 0, dpr = 1;
    var cells = [], markers = [], grown = false, rafId = 0, alive = true, visible = true, elapsed = 0, lastT = 0, lastFlick = 0, p = 0;

    function size() {
      var r = canvas.getBoundingClientRect();
      var w = Math.round(r.width) || +canvas.getAttribute('width') || 300;
      var h = Math.round(r.height) || +canvas.getAttribute('height') || 200;
      dpr = Math.min(win.devicePixelRatio || 1, 2);
      W = w; H = h;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.floor(w / pitch); rows = Math.floor(h / pitch);
    }

    function build() {
      var set = {}; cells = []; markers = [];
      function plot(x, y, seam) {
        x = Math.round(x); y = Math.round(y);
        if (x < 0 || y < 0 || x >= cols || y >= rows) return;
        var k = x + ',' + y;
        if (set[k]) { if (seam) set[k].seam = true; return; }
        var c = { x: x, y: y, seam: !!seam }; set[k] = c; cells.push(c);
      }
      var nf = [[0.10, 0.82], [0.32, 0.62], [0.50, 0.70], [0.70, 0.42], [0.88, 0.18]];
      var nodes = nf.map(function (f) { return [f[0] * cols, f[1] * rows]; });
      var thick = Math.max(1, Math.round(Math.min(cols, rows) * 0.03));
      for (var i = 0; i < nodes.length - 1; i++) {
        var a = nodes[i], b = nodes[i + 1], steps = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1])) * 2;
        for (var s = 0; s <= steps; s++) {
          var t = s / steps, x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t;
          for (var w = -thick; w <= thick; w++) { plot(x, y + w); plot(x + 0.5, y + w); }
        }
      }
      var hd = nodes[nodes.length - 1], A = Math.max(4, Math.round(rows * 0.16));   /* arrowhead */
      for (var dx = 0; dx <= A; dx++) for (var dy = 0; dy <= A - dx; dy++) plot(hd[0] - dx, hd[1] + dy, true);
      /* reveal order + colour: distance from the bottom-left origin, so it climbs the line */
      var ox = 0, oy = rows - 1, md = 0, ii;
      for (ii = 0; ii < cells.length; ii++) { cells[ii]._d = Math.hypot(cells[ii].x - ox, cells[ii].y - oy); if (cells[ii]._d > md) md = cells[ii]._d; }
      md = md || 1;
      for (ii = 0; ii < cells.length; ii++) {
        var c = cells[ii], rr = Math.random();
        c.seq = Math.min(1, Math.max(0, c._d / md + (Math.random() * 0.04 - 0.02)));
        c.ci = c.seam ? 4 : (rr < 0.34 ? 0 : rr < 0.6 ? 1 : rr < 0.82 ? 2 : 3);
        c.old = !c.seam && Math.random() < 0.10;
        c.a = 0.72 + Math.random() * 0.28;
      }
      /* inspector squares — climbing WAR projection; fewer on small canvases */
      var defs = [{ n: 0, label: '+1.2' }, { n: 1, label: '+3.8' }, { n: 3, label: '+6.4' }, { n: 4, label: '+9.7', peak: true }];
      var use = W >= 520 ? defs : W >= 230 ? [defs[1], defs[3]] : [];
      markers = use.map(function (o) {
        var nd = nodes[o.n];
        return { x: nd[0], y: nd[1], label: o.label, peak: !!o.peak, seq: Math.min(1, Math.hypot(nd[0] - ox, nd[1] - oy) / md + 0.02) };
      });
    }

    function drawMarkers() {
      if (!markers.length) return;
      var pal = opts.palette, s = Math.max(16, Math.round(opts.cell * 3.6));
      ctx.save();
      ctx.font = Math.max(8, Math.round(opts.cell * 1.7)) + 'px "JetBrains Mono", ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      for (var i = 0; i < markers.length; i++) {
        var m = markers[i]; if (m.seq > p) continue;
        var nx = m.x * pitch + opts.cell / 2, ny = m.y * pitch + opts.cell / 2;
        /* square sits ON the arrow, centred on the node (no leader line) */
        var bx = nx - s / 2, by = ny - s / 2;
        bx = Math.max(1, Math.min(W - s - 1, bx)); by = Math.max(1, Math.min(H - s - 1, by));
        ctx.fillStyle = 'rgba(0,0,0,.72)'; ctx.fillRect(Math.round(bx) + 1, Math.round(by) + 1, s - 1, s - 1);
        ctx.lineWidth = 1.5; ctx.strokeStyle = m.peak ? pal[0] : 'rgba(255,255,255,.92)';
        ctx.strokeRect(Math.round(bx) + 0.5, Math.round(by) + 0.5, s, s);
        var lab = m.label, lw = ctx.measureText(lab).width, lx = bx + s + 4;
        if (lx + lw > W - 2) lx = bx - 4 - lw;
        ctx.textAlign = 'left'; ctx.fillStyle = m.peak ? pal[0] : '#fff';
        ctx.fillText(lab, lx, by + s / 2);
      }
      ctx.restore();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      var pal = opts.palette;
      for (var i = 0; i < cells.length; i++) {
        var c = cells[i]; if (c.seq > p) continue;
        var lead = p - c.seq, tip = (!grown && lead >= 0 && lead < 0.05);
        ctx.globalAlpha = tip ? 1 : c.a;
        ctx.fillStyle = tip ? pal[4] : (c.old ? opts.old : pal[c.ci]);
        ctx.fillRect(c.x * pitch, c.y * pitch, opts.cell, opts.cell);
      }
      ctx.globalAlpha = 1;
      drawMarkers();
    }

    function flicker() {
      if (!opts.flicker || REDUCED) return;
      var n = cells.length; if (!n) return;
      var k = Math.max(1, (n * 0.05) | 0);
      for (var i = 0; i < k; i++) { var c = cells[(Math.random() * n) | 0]; if (c.seq > p) continue; c.a = rand(0.4, 1); if (Math.random() < 0.04 && !c.seam) c.old = !c.old; }
    }

    function loop() {
      if (!alive) return;
      rafId = raf(loop);
      var t = now();
      if (!visible) { lastT = 0; return; }              // offscreen: growth clock pauses
      var dt = lastT ? Math.min(t - lastT, 100) : 0; lastT = t;
      if (!grown) {
        elapsed += dt; p = Math.min(1, elapsed / opts.growth);
        if (p >= 1) grown = true;
        draw();
      } else if (t - lastFlick > 90) { flicker(); lastFlick = t; draw(); }
    }

    function start() {
      size(); build(); p = 0; grown = false; elapsed = 0; lastT = 0;
      if (REDUCED) { p = 1; grown = true; draw(); return; }
      draw();
      if (!rafId) loop();
    }

    function relayout() {
      var wasGrown = grown, wasP = p, wasElapsed = elapsed;
      size(); build();
      if (wasGrown || REDUCED) { p = 1; grown = true; } else { p = wasP; elapsed = wasElapsed; }
      draw();
      if (!REDUCED && !rafId) loop();
    }

    var rto, ro = null, io = null;
    function onResize() {
      if (!alive) return;
      clearTimeout(rto);
      rto = setTimeout(function () {
        if (!alive) return;
        var r = canvas.getBoundingClientRect();
        if (Math.round(r.width) !== W || Math.round(r.height) !== H) relayout();
      }, 150);
    }
    win.addEventListener('resize', onResize);
    if ('ResizeObserver' in win) { try { ro = new ResizeObserver(onResize); ro.observe(canvas); } catch (e) { ro = null; } }
    if ('IntersectionObserver' in win) {
      try { io = new IntersectionObserver(function (en) { visible = en.some(function (e) { return e.isIntersecting; }); }, { rootMargin: '100px' }); io.observe(canvas); } catch (e) { io = null; }
    }
    start();
    var api = {
      canvas: canvas, restart: start,
      stats: function () { return { cells: cells.length, grown: grown, p: p, cols: cols, rows: rows, visible: visible, w: W, h: H }; },
      destroy: function () {
        alive = false; clearTimeout(rto);
        win.removeEventListener('resize', onResize);
        if (ro) { try { ro.disconnect(); } catch (e) {} ro = null; }
        if (io) { try { io.disconnect(); } catch (e) {} io = null; }
        if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
        if (canvas.__dendrite === api) canvas.__dendrite = null;
      }
    };
    canvas.__dendrite = api;
    return api;
  };

  /* ---------------------------------------------------------------------
     marquee(el, opts) — duplicates .marquee__group so the CSS translateX(-50%) loop is seamless.
     el contains one .marquee__group (or bare .marquee__item children, which get wrapped).
     opts: {duration:'35s'}; data-marquee="35s" also sets the duration.
     --------------------------------------------------------------------- */
  DraftDB.marquee = function (el, opts) {
    el = toEl(el); if (!el) return;
    opts = opts || {};
    el.classList.add('marquee');
    var track = el.querySelector('.marquee__track');
    if (!track) {
      track = doc.createElement('div'); track.className = 'marquee__track';
      while (el.firstChild) track.appendChild(el.firstChild);
      el.appendChild(track);
    }
    var group = track.querySelector('.marquee__group');
    if (!group) {
      group = doc.createElement('div'); group.className = 'marquee__group';
      while (track.firstChild) group.appendChild(track.firstChild);
      track.appendChild(group);
    }
    $all('.marquee__group[data-clone]', track).forEach(function (g) { g.remove(); });
    var need = el.clientWidth || win.innerWidth;
    var items = $all('.marquee__item', group);
    var guard = 0;
    while (group.scrollWidth < need && items.length && guard++ < 10) {
      items.forEach(function (it) { group.appendChild(it.cloneNode(true)); });
    }
    var clone = group.cloneNode(true); clone.setAttribute('data-clone', '1'); clone.setAttribute('aria-hidden', 'true');
    track.appendChild(clone);
    if (opts.duration) el.style.setProperty('--marquee-duration', opts.duration);
    else if (el.dataset.marquee && /^\d/.test(el.dataset.marquee)) el.style.setProperty('--marquee-duration', el.dataset.marquee);
    return { track: track };
  };

  /* ---------------------------------------------------------------------
     accordion(el, opts) — .acc with .acc__item > .acc__head (button) + .acc__body > div
     opts: {single:true, initial:0|-1}. data-accordion="multi" → several open; data-collapsible="true" → active row can close.
     Returns {open(i), close(i)}
     --------------------------------------------------------------------- */
  DraftDB.accordion = function (el, opts) {
    el = toEl(el); if (!el) return;
    opts = Object.assign({ single: true, initial: 0 }, opts || {});
    if (el.dataset.accordion === 'multi') opts.single = false;
    var items = $all('.acc__item', el);
    function open(item) { item.classList.add('is-active'); var h = item.querySelector('.acc__head'); if (h) h.setAttribute('aria-expanded', 'true'); }
    function close(item) { item.classList.remove('is-active'); var h = item.querySelector('.acc__head'); if (h) h.setAttribute('aria-expanded', 'false'); }
    items.forEach(function (item) {
      var head = item.querySelector('.acc__head'); if (!head) return;
      head.setAttribute('aria-expanded', item.classList.contains('is-active') ? 'true' : 'false');
      head.addEventListener('click', function () {
        var isOpen = item.classList.contains('is-active');
        if (opts.single) items.forEach(function (o) { if (o !== item) close(o); });
        if (isOpen && (!opts.single || el.dataset.collapsible === 'true')) close(item); else open(item);
      });
    });
    if (!items.some(function (i) { return i.classList.contains('is-active'); }) && opts.initial >= 0 && items[opts.initial]) open(items[opts.initial]);
    return { open: function (i) { if (opts.single) items.forEach(close); if (items[i]) open(items[i]); }, close: function (i) { if (items[i]) close(items[i]); } };
  };

  /* ---------------------------------------------------------------------
     initNav() — mobile menu toggle (< 960px), active link, scrolled state
     --------------------------------------------------------------------- */
  DraftDB.initNav = function () {
    var nav = doc.querySelector('.nav'); if (!nav) return;
    var btn = nav.querySelector('.nav__menu-btn');
    var links = nav.querySelector('.nav__links');
    if (btn && links) {
      btn.setAttribute('aria-expanded', 'false');
      btn.addEventListener('click', function () {
        var open = nav.classList.toggle('is-open');
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
      links.addEventListener('click', function (e) { if (e.target.tagName === 'A') { nav.classList.remove('is-open'); btn.setAttribute('aria-expanded', 'false'); } });
      win.addEventListener('resize', function () { if (win.innerWidth >= 960) nav.classList.remove('is-open'); });
    }
    var here = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
    $all('.nav__links a', nav).forEach(function (a) {
      var href = (a.getAttribute('href') || '').split('#')[0].toLowerCase();
      if (href && href === here) a.classList.add('is-active');
    });
    var onScroll = function () { nav.classList.toggle('is-scrolled', (win.scrollY || doc.documentElement.scrollTop) > 8); };
    win.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  };

  /* ---------------------------------------------------------------------
     init(root) — auto wiring for data-* hooks (runs on DOMContentLoaded)
     --------------------------------------------------------------------- */
  DraftDB.init = function (root) {
    root = toEl(root) || doc;
    DraftDB.initNav();
    DraftDB.smoothScroll();
    $all('[data-decode]', root).forEach(function (el) {
      var o = {};
      if (el.dataset.decodeDelay) o.delay = +el.dataset.decodeDelay;
      if (el.dataset.decodeDuration) o.duration = +el.dataset.decodeDuration;
      DraftDB.decodeText(el, o);
    });
    $all('[data-roll]', root).forEach(function (el) { DraftDB.rollNumber(el, { trigger: el.dataset.rollTrigger || 'visible' }); });
    $all('canvas[data-dendrite]', root).forEach(function (c) { DraftDB.sprout(c); });
    $all('[data-marquee]', root).forEach(function (el) { DraftDB.marquee(el); });
    $all('[data-accordion], .acc', root).forEach(function (el) { if (!el.__acc) { el.__acc = true; DraftDB.accordion(el); } });
    DraftDB.reveal(root);
    doc.documentElement.classList.add('js-ready');
  };

  win.DraftDB = DraftDB;
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', function () { DraftDB.init(); });
  else DraftDB.init();
})(window, document);
