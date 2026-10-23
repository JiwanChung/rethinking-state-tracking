(function () {
  'use strict';

  const FS_URL = 'static/data/fs_models.json';
  const Q_URL = 'static/data/q_models.json';
  const ROLES = ['affine', 'state'];
  const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const NARROW_VIEWPORT = 700;
  const VERIFY_TOL = 1e-9;

  const COLOR = {
    affine: '#c0392b', state: '#2b6cb0',
    grid: '#f0f0f0', axis: '#cfcfcf',
    text: '#2a2a2a', muted: '#6b6b6b', ink: '#0f0f0f', clean: '#9b9b9b',
  };
  const FAMILY = 'Inter, "Helvetica Neue", Arial, sans-serif';
  const FONT = '12px ' + FAMILY;
  const LABEL_FONT = '13px ' + FAMILY;

  function need(root, selector) {
    const el = root.querySelector(selector);
    if (!el) throw new Error('missing element: ' + selector);
    return el;
  }
  function setText(root, role, text) {
    need(root, '[data-out="' + role + '"]').textContent = text;
  }
  function line(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
  function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
  function setPlayIcon(btn, playing) {
    need(btn, 'use').setAttribute('href', playing ? '#i-pause' : '#i-play');
    btn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }
  function onVisible(el, fn) {
    const obs = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); fn(); }
    }, { threshold: 0.35 });
    obs.observe(el);
  }
  function View(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.aspectWide = Number(canvas.dataset.aspect);
    if (!(this.aspectWide > 0)) throw new Error('canvas needs a positive data-aspect: ' + canvas.outerHTML);
    this.aspectNarrow = canvas.dataset.aspectNarrow === undefined ? this.aspectWide : Number(canvas.dataset.aspectNarrow);
    if (!(this.aspectNarrow > 0)) throw new Error('data-aspect-narrow must be positive: ' + canvas.outerHTML);
    this.resize();
  }
  View.prototype.resize = function () {
    const aspect = window.innerWidth < NARROW_VIEWPORT ? this.aspectNarrow : this.aspectWide;
    const w = this.canvas.clientWidth, h = Math.round(w / aspect), dpr = window.devicePixelRatio;
    this.canvas.style.height = h + 'px';
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
  };

  const SUB = ['₀', '₁', '₂'];
  const SCRIPT_U = '\u{1D4B0}';
  const SIGMOID = (v) => 1 / (1 + Math.exp(-v));
  function arrow(ctx, x0, y0, x1, y1) {
    line(ctx, x0, y0, x1, y1);
    const a = Math.atan2(y1 - y0, x1 - x0), L = 8;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - L * Math.cos(a - 0.4), y1 - L * Math.sin(a - 0.4));
    ctx.lineTo(x1 - L * Math.cos(a + 0.4), y1 - L * Math.sin(a + 0.4));
    ctx.closePath();
    ctx.fill();
  }
  function barycentric(p, a, b, c) {
    const det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    const wa = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det;
    const wb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det;
    return [wa, wb, 1 - wa - wb];
  }
  function matrix(rows, n, m, what) {
    if (rows.length !== n || rows.some((r) => r.length !== m)) throw new Error(what + ': expected ' + n + 'x' + m);
    const a = new Float64Array(n * m);
    rows.forEach((r, i) => a.set(r, i * m));
    return a;
  }
  function matvec(W, v, n, m, out, o) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < m; j++) s += W[i * m + j] * v[j];
      out[o + i] = s;
    }
  }

  function StateMap(spec, word) {
    const p = spec.params;
    this.kind = p.kind;
    this.word = word;
    this.dim = spec.c[0].length;
    const tok = (obj, x, len, what) => {
      const v = obj[String(x)];
      if (!v || v.length !== len) throw new Error(spec.label + ': missing ' + what + ' for input ' + x);
      return Float64Array.from(v);
    };
    if (this.kind === 'tanh') {
      const d = this.d = p.bhh.length;
      this.Whh = matrix(p.Whh, d, d, 'Whh');
      this.bhh = Float64Array.from(p.bhh);
      this.pre = {};
      for (const x of word) this.pre[x] = tok(p.pre, x, d, 'pre');
    } else if (this.kind === 'state_gated') {
      const d = this.d = p.bh.length;
      this.Ug = matrix(p.Ug, d, d, 'Ug');
      this.Wh = matrix(p.Wh, d, d, 'Wh');
      this.bh = Float64Array.from(p.bh);
      this.gx = {};
      this.wx = {};
      for (const x of word) { this.gx[x] = tok(p.gx, x, d, 'gx'); this.wx[x] = tok(p.wx, x, d, 'wx'); }
    } else {
      throw new Error(spec.label + ': no page F_s for model kind ' + this.kind);
    }
    if (this.dim !== this.d) throw new Error(spec.label + ': state size does not match its parameters');
    this.buf = new Float64Array(2 * this.d);
    this.plane = { mean: Float64Array.from(spec.plane.mean), bx: Float64Array.from(spec.plane.bx), by: Float64Array.from(spec.plane.by) };
  }
  StateMap.prototype.step = function (s, x) {
    const d = this.d, g = this.buf, out = new Float64Array(this.dim);
    if (this.kind === 'tanh') {
      matvec(this.Whh, s, d, d, g, 0);
      for (let i = 0; i < d; i++) out[i] = Math.tanh(g[i] + this.bhh[i] + this.pre[x][i]);
    } else {
      matvec(this.Ug, s, d, d, g, 0);
      matvec(this.Wh, s, d, d, g, d);
      for (let i = 0; i < d; i++) out[i] = SIGMOID(this.gx[x][i] + g[i]) * (g[d + i] + this.bh[i]) + this.wx[x][i];
    }
    return out;
  };
  StateMap.prototype.fs = function (s) {
    for (const x of this.word) s = this.step(s, x);
    return s;
  };
  StateMap.prototype.project = function (s) {
    const { mean, bx, by } = this.plane;
    let a = 0, b = 0;
    for (let j = 0; j < this.dim; j++) { const v = s[j] - mean[j]; a += v * bx[j]; b += v * by[j]; }
    return [a, b];
  };

  function initTheorem(root, fsData) {
    const scrub = need(root, '[data-role=scrub]');
    const playBtn = need(root, '[data-role=play]');
    const N = fsData.passes;
    if (Number(scrub.max) !== N) throw new Error('scrubber range must match the ' + N + ' passes in the data');
    const G = 2;
    const MAX_DELTA = 1.2;
    const PASS_MS = 700;
    const PAD = { L: 28, R: 28, T: 40, B: 40 };
    const NODE_R = 11;
    const ROLE_OF = { affine: 'affine', state: 'state' };
    const views = {}, pick = {}, picked = {}, maps = {};
    const s = { n: 0, raf: 0, last: 0, acc: 0, dragging: false, lambda: fsData.default_weights.slice(), runs: {}, geo: {} };

    const specs = { affine: [], state: [] };
    for (const spec of fsData.models) {
      if (!ROLE_OF[spec.role]) throw new Error(spec.label + ': unknown role ' + spec.role);
      specs[spec.role].push(spec);
      if (spec.nodes.length !== N + 1) throw new Error(spec.label + ': expected ' + (N + 1) + ' passes of nodes');
    }
    for (const role of ROLES) if (!specs[role].length) throw new Error('no ' + role + ' model in the data');

    for (const spec of specs.state) {
      const map = new StateMap(spec, fsData.word);
      let cs = spec.c.map((v) => Float64Array.from(v));
      const nodes = [];
      for (let n = 0; n <= N; n++) {
        if (n) cs = cs.map((v) => map.fs(v));
        nodes.push(cs.map((v) => map.project(v)));
      }
      const e0 = edgeOf(spec.nodes[0]);
      let err = 0;
      nodes.forEach((row, n) => row.forEach((q, k) => { err = Math.max(err, Math.hypot(q[0] - spec.nodes[n][k][0], q[1] - spec.nodes[n][k][1]) / e0); }));
      const ref = trajectory(map, spec, spec.reference.weights);
      ref.forEach((q, n) => { err = Math.max(err, Math.hypot(q[0] - spec.reference.pert[n][0], q[1] - spec.reference.pert[n][1]) / e0); });
      if (!(err <= VERIFY_TOL)) throw new Error(spec.label + ': page F_s differs from the exported one by ' + err.toExponential(2) + ' (relative to the triangle)');
      maps[spec.key] = map;
    }
    function edgeOf(tri) {
      let e = Infinity;
      for (let a = 0; a < 3; a++) for (let b = a + 1; b < 3; b++) e = Math.min(e, Math.hypot(tri[a][0] - tri[b][0], tri[a][1] - tri[b][1]));
      return e;
    }
    function trajectory(map, spec, lam) {
      let p = new Float64Array(map.dim);
      for (let k = 0; k < 3; k++) for (let j = 0; j < map.dim; j++) p[j] += lam[k] * spec.c[k][j];
      const out = [map.project(p)];
      for (let n = 1; n <= N; n++) { p = map.fs(p); out.push(map.project(p)); }
      return out;
    }
    function simulate(role) {
      const spec = picked[role], lam = s.lambda, raw = spec.nodes;
      const pertRaw = spec.role === 'affine'
        ? raw.map((row) => [0, 1].map((i) => lam[0] * row[0][i] + lam[1] * row[1][i] + lam[2] * row[2][i]))
        : trajectory(maps[spec.key], spec, lam);
      const center = raw.map((row) => [0, 1].map((i) => (row[0][i] + row[1][i] + row[2][i]) / 3));
      const nodes = raw.map((row, n) => row.map((q) => [q[0] - center[n][0], q[1] - center[n][1]]));
      const pert = pertRaw.map((q, n) => [q[0] - center[n][0], q[1] - center[n][1]]);
      return { nodes, pert, weights: pert.map((q, n) => barycentric(q, nodes[n][0], nodes[n][1], nodes[n][2])) };
    }
    function rebuild() { for (const role of ROLES) s.runs[role] = simulate(role); }

    function fitBox(role) {
      const spec = picked[role], keep = s.lambda;
      s.lambda = fsData.default_weights.slice();
      const run = simulate(role);
      s.lambda = keep;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      const add = (p) => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); };
      run.nodes.forEach((row) => row.forEach(add));
      run.pert.forEach(add);
      s.geo[role] = { box: { x0, x1, y0, y1 }, edge: edgeOf(spec.nodes[0]) };
    }
    function layout(role) {
      const view = views[role], G2 = s.geo[role], b = G2.box;
      const sc = Math.min((view.w - PAD.L - PAD.R) / (b.x1 - b.x0), (view.h - PAD.T - PAD.B) / (b.y1 - b.y0));
      const cx = PAD.L + (view.w - PAD.L - PAD.R) / 2, cy = PAD.T + (view.h - PAD.T - PAD.B) / 2;
      const mx = (b.x0 + b.x1) / 2, my = (b.y0 + b.y1) / 2;
      G2.X = (u) => cx + (u - mx) * sc;
      G2.Y = (v) => cy - (v - my) * sc;
      G2.Ui = (px) => mx + (px - cx) / sc;
      G2.Vi = (py) => my - (py - cy) / sc;
    }

    function setN(n) { s.n = n; scrub.value = String(n); render(); }
    function stop() { if (s.raf) { cancelAnimationFrame(s.raf); s.raf = 0; } setPlayIcon(playBtn, false); }
    function frame(now) {
      s.acc += (now - s.last) / PASS_MS;
      s.last = now;
      if (s.acc >= 1) { s.acc -= 1; setN(Math.min(N, s.n + 1)); }
      if (s.n >= N) { stop(); return; }
      s.raf = requestAnimationFrame(frame);
    }
    function play() {
      if (s.n >= N) setN(0);
      setPlayIcon(playBtn, true);
      s.acc = 0;
      s.last = performance.now();
      s.raf = requestAnimationFrame(frame);
    }

    const pct = (w) => Math.round(100 * w) + 0 + '%';  // + 0 turns -0 into 0
    function polyline(ctx, pts, X, Y, close) {
      ctx.beginPath();
      pts.forEach((q, i) => (i ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1]))));
      if (close) ctx.closePath();
      ctx.stroke();
    }
    function renderPanel(role) {
      const view = views[role], G2 = s.geo[role], ctx = view.ctx, run = s.runs[role], n = s.n;
      const X = G2.X, Y = G2.Y, cp = run.nodes[n], pp = run.pert[n], w = run.weights[n];
      ctx.clearRect(0, 0, view.w, view.h);
      ctx.font = FONT;
      ctx.fillStyle = COLOR.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('symbolic subspace ' + SCRIPT_U, 8, 8);
      if (n > 0) {
        const c0 = run.nodes[0];
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = COLOR.axis;
        polyline(ctx, c0, X, Y, true);
        ctx.strokeStyle = COLOR.clean;
        for (const q of c0) { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), NODE_R, 0, 2 * Math.PI); ctx.stroke(); }
        ctx.strokeStyle = COLOR[role];
        ctx.beginPath();
        ctx.arc(X(run.pert[0][0]), Y(run.pert[0][1]), 7, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = COLOR.clean;
        for (let g = 0; g < 3; g++) polyline(ctx, run.nodes.slice(0, n + 1).map((row) => row[g]), X, Y, false);
        ctx.strokeStyle = COLOR[role];
        polyline(ctx, run.pert.slice(0, n + 1), X, Y, false);
      }
      ctx.strokeStyle = COLOR.axis;
      ctx.lineWidth = 1;
      polyline(ctx, cp, X, Y, true);
      const px = X(pp[0]), py = Y(pp[1]);
      if (n === 0) {
        const gx = X(cp[G][0]), gy = Y(cp[G][1]), len = Math.hypot(px - gx, py - gy);
        ctx.strokeStyle = COLOR.ink;
        ctx.fillStyle = COLOR.ink;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 3]);
        if (len > NODE_R + 18) {
          arrow(ctx, gx + (px - gx) * NODE_R / len, gy + (py - gy) * NODE_R / len, px - (px - gx) * 8 / len, py - (py - gy) * 8 / len);
        }
        ctx.setLineDash([]);
        const a = Math.atan2(py - gy, px - gx);
        ctx.font = 'italic ' + LABEL_FONT;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('δ', (gx + px) / 2 + 12 * Math.sin(a), (gy + py) / 2 - 12 * Math.cos(a));
      }
      for (let i = 0; i < 3; i++) {
        const x = X(cp[i][0]), y = Y(cp[i][1]);
        ctx.beginPath();
        ctx.arc(x, y, NODE_R, 0, 2 * Math.PI);
        ctx.fillStyle = i === G ? COLOR.ink : '#ffffff';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = COLOR.ink;
        ctx.stroke();
        ctx.font = FONT;
        ctx.fillStyle = i === G ? '#ffffff' : COLOR.ink;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('c' + SUB[i], x, y);
      }
      ctx.fillStyle = COLOR[role];
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.font = LABEL_FONT;
      ctx.fillStyle = COLOR.text;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('toward c₀: ' + pct(w[0]) + '    toward c₁: ' + pct(w[1]), 8, view.h - 8);
    }
    function render() {
      ROLES.forEach(renderPanel);
      setText(root, 'n', String(s.n));
    }

    function place(role, e) {
      const G2 = s.geo[role], r = views[role].canvas.getBoundingClientRect(), cp = s.runs[role].nodes[0];
      let d = [G2.Ui(e.clientX - r.left) - cp[G][0], G2.Vi(e.clientY - r.top) - cp[G][1]];
      const len = Math.hypot(d[0], d[1]);
      if (len > MAX_DELTA * G2.edge) d = [d[0] * MAX_DELTA * G2.edge / len, d[1] * MAX_DELTA * G2.edge / len];
      s.lambda = barycentric([cp[G][0] + d[0], cp[G][1] + d[1]], cp[0], cp[1], cp[2]);
      rebuild();
      setN(0);
    }
    function choose(role, spec) {
      picked[role] = spec;
      for (const b of pick[role].children) b.classList.toggle('is-on', b.dataset.key === spec.key);
      fitBox(role);
      layout(role);
    }
    for (const role of ROLES) {
      views[role] = new View(need(root, 'canvas[data-panel=' + role + ']'));
      pick[role] = need(root, '[data-role=pick][data-panel=' + role + ']');
      for (const spec of specs[role]) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'toggle';
        b.dataset.key = spec.key;
        b.textContent = spec.label;
        b.addEventListener('click', () => { stop(); choose(role, spec); rebuild(); setN(0); if (!REDUCED_MOTION) play(); });
        pick[role].appendChild(b);
      }
      const c = views[role].canvas;
      c.addEventListener('pointerdown', (e) => { stop(); s.dragging = true; c.setPointerCapture(e.pointerId); place(role, e); });
      c.addEventListener('pointermove', (e) => { if (s.dragging) place(role, e); });
      c.addEventListener('pointerup', () => { if (!s.dragging) return; s.dragging = false; if (!REDUCED_MOTION) play(); });
      c.addEventListener('pointercancel', () => { s.dragging = false; });
    }
    scrub.addEventListener('input', () => { stop(); setN(Number(scrub.value)); });
    need(root, '[data-role=step]').addEventListener('click', () => { stop(); if (s.n < N) setN(s.n + 1); });
    playBtn.addEventListener('click', () => { if (s.raf) stop(); else play(); });
    need(root, '[data-role=reset]').addEventListener('click', () => { stop(); setN(0); });
    window.addEventListener('resize', () => {
      for (const role of ROLES) { views[role].resize(); layout(role); }
      render();
    });
    document.fonts.ready.then(render);

    for (const role of ROLES) choose(role, specs[role][0]);
    rebuild();
    setN(REDUCED_MOTION ? N : 0);
    if (!REDUCED_MOTION) onVisible(root, play);
  }

  const SHADES = { affine: ['#e8a097', '#c0392b', '#6e1e15'], state: ['#98bde6', '#2b6cb0', '#163b63'] };
  function bytes(b64) {
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function initCorollary(root, qData) {
    const T = qData.rollouts.T, n = qData.rollouts.n, TAU = qData.tau, FR = qData.frames;
    const scrub = need(root, '[data-role=scrub]');
    const playBtn = need(root, '[data-role=play]');
    if (Number(scrub.max) !== T) throw new Error('scrubber range must match the rollout length ' + T);
    const F = FR.steps.length;
    const labels = bytes(FR.labels);
    if (labels.length !== F * n) throw new Error('state labels: expected ' + F + ' frames of ' + n + ' runs');
    const frameAt = new Int32Array(T + 1);
    for (let t = 1, f = -1; t <= T; t++) { while (f + 1 < F && FR.steps[f + 1] <= t) f++; frameAt[t] = f; }
    if (frameAt[1] !== 0) throw new Error('the first stored frame must be step 1');
    const specs = { affine: [], state: [] };
    for (const spec of qData.models) {
      if (!specs[spec.role]) throw new Error(spec.label + ': unknown role ' + spec.role);
      if (spec.q.length !== T) throw new Error(spec.label + ': q must have ' + T + ' steps');
      if (spec.nodes.length !== F) throw new Error(spec.label + ': expected ' + F + ' frames of centroids');
      spec.posBytes = new Int8Array(bytes(spec.pos).buffer);
      if (spec.posBytes.length !== F * n * 2) throw new Error(spec.label + ': expected ' + F + ' frames of ' + n + ' runs');
      specs[spec.role].push(spec);
    }
    for (const role of ROLES) if (!specs[role].length) throw new Error('no ' + role + ' model in the Corollary 1 data');
    const STEPS_PER_SECOND = 25;
    const MARGIN = { L: 54, R: 16 };
    const NODE_R = 8;
    const views = {}, pick = {}, picked = {};
    const qview = new View(need(root, 'canvas[data-role=plot]'));
    const s = { t: 1, playing: false, raf: 0, last: 0, acc: 0 };

    function setT(t) { s.t = t; scrub.value = String(t); render(); }
    function frame(now) {
      s.acc += STEPS_PER_SECOND * (now - s.last) / 1000;
      s.last = now;
      const k = Math.floor(s.acc);
      s.acc -= k;
      if (k > 0) setT(Math.min(T, s.t + k));
      if (s.t >= T) { stop(); return; }
      s.raf = requestAnimationFrame(frame);
    }
    function play() {
      if (s.t >= T) setT(1);
      s.playing = true;
      setPlayIcon(playBtn, true);
      s.acc = 0;
      s.last = performance.now();
      s.raf = requestAnimationFrame(frame);
    }
    function stop() { s.playing = false; if (s.raf) { cancelAnimationFrame(s.raf); s.raf = 0; } setPlayIcon(playBtn, false); }

    const fmtQ = (q) => (q < 10 ? q.toFixed(2) : q.toFixed(1));
    function renderCloud(role) {
      const view = views[role], ctx = view.ctx, spec = picked[role], f = frameAt[s.t];
      const half = Math.min(view.w, view.h) / 2 - 6, cx = view.w / 2, cy = view.h / 2;
      const X = (u) => cx + u * half, Y = (v) => cy - v * half;
      ctx.clearRect(0, 0, view.w, view.h);
      const P = spec.posBytes, o = f * n, shade = SHADES[role];
      let outside = 0;
      ctx.globalAlpha = 0.6;
      for (let i = 0; i < n; i++) {
        const px = P[(o + i) * 2], py = P[(o + i) * 2 + 1];
        if (px === FR.outside) { outside++; continue; }
        ctx.fillStyle = shade[labels[o + i]];
        ctx.beginPath();
        ctx.arc(X(px / 127), Y(py / 127), 2.6, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      spec.nodes[f].forEach((q, g) => {
        const x = X(q[0]), y = Y(q[1]);
        ctx.beginPath();
        ctx.arc(x, y, NODE_R, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = shade[g];
        ctx.stroke();
        ctx.font = '10px ' + FAMILY;
        ctx.fillStyle = COLOR.ink;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('c' + SUB[g], x, y);
      });
      ctx.font = FONT;
      ctx.fillStyle = COLOR.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('readout plane', 8, 8);
      if (outside > 0) {
        ctx.textAlign = 'right';
        ctx.fillText(outside + ' runs beyond the view', view.w - 8, 8);
      }
      const q = spec.q[s.t - 1];
      ctx.font = LABEL_FONT;
      ctx.fillStyle = q >= TAU ? COLOR.ink : COLOR.text;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('q(t) = ' + fmtQ(q), 8, view.h - 8);
    }

    const xOf = (view, t) => MARGIN.L + (t - 1) / (T - 1) * (view.w - MARGIN.L - MARGIN.R);
    function renderPlot() {
      const view = qview, ctx = view.ctx, t = s.t;
      const top = 8, bottom = view.h - 36, x0 = MARGIN.L, x1 = view.w - MARGIN.R;
      const Q_LO = -2, Q_HI = 2;
      const yQ = (q) => top + (Q_HI - clamp(Math.log10(q), Q_LO, Q_HI)) / (Q_HI - Q_LO) * (bottom - top);
      ctx.clearRect(0, 0, view.w, view.h);
      ctx.font = FONT;
      ctx.lineWidth = 1;
      ctx.strokeStyle = COLOR.grid;
      ctx.fillStyle = COLOR.muted;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let e = Q_LO; e <= Q_HI; e++) {
        line(ctx, x0, yQ(Math.pow(10, e)), x1, yQ(Math.pow(10, e)));
        ctx.fillText(String(Math.pow(10, e)), x0 - 6, yQ(Math.pow(10, e)));
      }
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (const tick of [1, 50, 100, 150, 200]) {
        line(ctx, xOf(view, tick), top, xOf(view, tick), bottom);
        ctx.fillText(String(tick), xOf(view, tick), bottom + 5);
      }
      ctx.fillText('step t', (x0 + x1) / 2, bottom + 20);
      ctx.strokeStyle = COLOR.axis;
      ctx.strokeRect(x0 + 0.5, top + 0.5, x1 - x0 - 1, bottom - top - 1);
      ctx.save();
      ctx.translate(12, (top + bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle';
      ctx.fillText('q(t)  (log)', 0, 0);
      ctx.restore();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = COLOR.muted;
      line(ctx, x0, yQ(TAU), x1, yQ(TAU));
      ctx.setLineDash([]);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText('τ = ½', x1 - 6, yQ(TAU) - 3);
      for (const role of ROLES) {
        const spec = picked[role], q = spec.q;
        ctx.strokeStyle = COLOR[role];
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 1; i < t; i++) {
          const px = xOf(view, i + 1), py = yQ(q[i]);
          if (i === 1) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.fillStyle = COLOR[role];
        ctx.textAlign = 'left';
        if (spec.regime === 'climb' && spec.t_cross <= t) {
          const px = xOf(view, spec.t_cross), py = yQ(q[spec.t_cross - 1]);
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, 2 * Math.PI);
          ctx.fill();
          ctx.textBaseline = 'top';
          const text = spec.label + ' crosses ½ at t = ' + spec.t_cross;
          if (px + 6 + ctx.measureText(text).width > x1 - 4) { ctx.textAlign = 'right'; ctx.fillText(text, px - 6, py + 5); }
          else ctx.fillText(text, px + 6, py + 5);
        } else if (spec.regime === 'saturation' && t > 1) {
          ctx.textBaseline = 'bottom';
          ctx.fillText(spec.label + ': above ½ from the start', x0 + 6, yQ(Math.max(q[1], TAU)) - 6);
        }
      }
      if (t > 1 && t < T) {
        ctx.strokeStyle = COLOR.muted;
        ctx.lineWidth = 1;
        line(ctx, xOf(view, t), top, xOf(view, t), bottom);
      }
    }

    function render() {
      ROLES.forEach(renderCloud);
      renderPlot();
      setText(root, 't', String(s.t));
    }
    function choose(role, spec) {
      picked[role] = spec;
      for (const b of pick[role].children) b.classList.toggle('is-on', b.dataset.key === spec.key);
    }
    for (const role of ROLES) {
      views[role] = new View(need(root, 'canvas[data-role=cloud][data-panel=' + role + ']'));
      pick[role] = need(root, '[data-role=pick][data-panel=' + role + ']');
      for (const spec of specs[role]) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'toggle';
        b.dataset.key = spec.key;
        b.textContent = spec.label;
        b.addEventListener('click', () => { choose(role, spec); render(); });
        pick[role].appendChild(b);
      }
    }
    scrub.addEventListener('input', () => { stop(); setT(Number(scrub.value)); });
    playBtn.addEventListener('click', () => { if (s.playing) stop(); else play(); });
    need(root, '[data-role=reset]').addEventListener('click', () => { stop(); setT(1); });
    window.addEventListener('resize', () => {
      for (const role of ROLES) views[role].resize();
      qview.resize();
      render();
    });
    document.fonts.ready.then(render);
    for (const role of ROLES) choose(role, specs[role][0]);
    setT(REDUCED_MOTION ? T : 1);
    if (!REDUCED_MOTION) onVisible(root, play);
  }

  function fail(err) {
    for (const id of ['#widget-theorem', '#widget-corollary']) {
      const root = need(document, id), box = document.createElement('p');
      box.className = 'chart-error';
      box.textContent = 'This figure could not be drawn: ' + err.message;
      root.replaceChildren(box);
    }
    throw err;
  }

  async function load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('could not load ' + url + ' (HTTP ' + res.status + ')');
    return res.json();
  }
  async function init() {
    const [fsData, qData] = await Promise.all([load(FS_URL), load(Q_URL)]);
    initTheorem(need(document, '#widget-theorem'), fsData);
    initCorollary(need(document, '#widget-corollary'), qData);
  }
  function start() { init().catch(fail); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
