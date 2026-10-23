/* Live versions of two paper figures, drawn from the exported data in
 * static/data/ (scripts/export_page_data.py in the project repo):
 *   Fig 2  fig2_q_over_rollouts.json  q(t) and R(t) over rollouts, six models
 *   Fig 4  fig4_q_vs_t_over_mp.json   q_t against t / mp for the 107 trained runs in the correlation
 * Axis scales, limits, ticks, reference lines and colours come from the JSON,
 * which records them from the drawn paper figure. Anything the code does not
 * know how to draw raises instead of being approximated.
 */
(function () {
  'use strict';

  const FAMILY = 'Inter, "Helvetica Neue", Arial, sans-serif';
  const FONT = '12px ' + FAMILY;
  const SMALL = '10px ' + FAMILY;
  const TITLE = '500 13.5px ' + FAMILY;
  const INK = '#0f0f0f', MUTED = '#6b6b6b', GRID = '#efefef', FRAME = '#bdbdbd', REF = '#555555';
  const NARROW_VIEWPORT = 700;

  function need(root, selector) {
    const el = root.querySelector(selector);
    if (!el) throw new Error('missing element: ' + selector);
    return el;
  }

  async function loadJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('could not load ' + url + ' (HTTP ' + res.status + ')');
    return res.json();
  }

  // Shows the failure in place of the chart, then rethrows so it also reaches the console.
  function fail(chart, err) {
    const box = document.createElement('p');
    box.className = 'chart-error';
    box.textContent = 'This figure could not be drawn: ' + err.message;
    chart.replaceChildren(box);
    throw err;
  }

  function logAxis(spec, what) {
    if (spec !== 'log') throw new Error(what + ': only log scales are supported, got ' + spec);
  }
  function logScale(d0, d1, r0, r1) {
    const l0 = Math.log10(d0), l1 = Math.log10(d1);
    const f = (v) => r0 + (Math.log10(v) - l0) / (l1 - l0) * (r1 - r0);
    f.inv = (p) => Math.pow(10, l0 + (p - r0) / (r1 - r0) * (l1 - l0));
    return f;
  }

  function sizeCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio;
    canvas.style.height = h + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function line(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }

  // Polyline over (xs, ys). A null value, or a non-positive one on a log axis
  // (which matplotlib masks the same way in the paper figure), breaks the line.
  function polyline(ctx, xs, ys, sx, sy) {
    let pen = false;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const y = ys[i];
      if (y === null || !(y > 0)) { pen = false; continue; }
      const X = sx(xs[i]), Y = sy(y);
      if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
    }
    ctx.stroke();
  }

  // Filled band between lo and hi, in runs where both bounds are drawable.
  function band(ctx, xs, lo, hi, sx, sy) {
    let i = 0;
    while (i < xs.length) {
      while (i < xs.length && !(lo[i] > 0 && hi[i] > 0)) i++;
      const start = i;
      while (i < xs.length && lo[i] > 0 && hi[i] > 0) i++;
      if (i - start < 2) continue;
      ctx.beginPath();
      for (let k = start; k < i; k++) { if (k === start) ctx.moveTo(sx(xs[k]), sy(hi[k])); else ctx.lineTo(sx(xs[k]), sy(hi[k])); }
      for (let k = i - 1; k >= start; k--) ctx.lineTo(sx(xs[k]), sy(lo[k]));
      ctx.closePath();
      ctx.fill();
    }
  }

  // Tick labels: plain numbers, or 10^k written with a raised exponent.
  function tickText(ctx, label, x, y, align) {
    const m = /^\$\\mathdefault\{10\^\{(-?\d+)\}\}\$$/.exec(label);
    if (!m) {
      const plain = /^\$\\mathbf\{(.+)\}\$$/.exec(label);
      if (plain) { ctx.font = '600 ' + FONT; ctx.textAlign = align; ctx.fillText(plain[1], x, y); ctx.font = FONT; return; }
      if (/[$\\]/.test(label)) throw new Error('unsupported tick label: ' + label);
      ctx.textAlign = align;
      ctx.fillText(label, x, y);
      return;
    }
    const exp = m[1].replace('-', '−');
    ctx.font = SMALL;
    const ew = ctx.measureText(exp).width;
    ctx.font = FONT;
    const bw = ctx.measureText('10').width;
    const left = align === 'right' ? x - bw - ew : align === 'center' ? x - (bw + ew) / 2 : x;
    ctx.textAlign = 'left';
    ctx.fillText('10', left, y);
    ctx.font = SMALL;
    ctx.fillText(exp, left + bw, y - 5);
    ctx.font = FONT;
  }

  // Canvas text made of runs, each plain, raised (sup) or lowered (sub).
  function richWidth(ctx, runs) {
    let w = 0;
    runs.forEach((r) => { ctx.font = r.sup || r.sub ? SMALL : FONT; w += ctx.measureText(r.t).width; });
    ctx.font = FONT;
    return w;
  }
  function drawRich(ctx, runs, x, y) {
    ctx.textAlign = 'left';
    runs.forEach((r) => {
      ctx.font = r.sup || r.sub ? SMALL : FONT;
      ctx.fillText(r.t, x, y + (r.sup ? -5 : r.sub ? 3 : 0));
      x += ctx.measureText(r.t).width;
    });
    ctx.font = FONT;
  }
  function pRuns(p) {
    const e = Math.floor(Math.log10(p)), mant = p / Math.pow(10, e);
    return [{ t: 'p = ' + mant.toFixed(1) + ' × 10' }, { t: String(e).replace('-', '−'), sup: true }];
  }

  function niceT(t) { return String(Math.round(t)); }
  function fmt(v) {
    if (v === null) return 'n/a';
    if (v !== 0 && (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2)) {
      const e = Math.floor(Math.log10(Math.abs(v)));
      return (v / Math.pow(10, e)).toFixed(2) + ' × 10<sup>' + String(e).replace('-', '−') + '</sup>';
    }
    return v.toPrecision(3);
  }

  function placeTip(tip, chart, x, y, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const cw = chart.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 14, top = y - th - 10;
    if (left + tw > cw) left = x - tw - 14;
    if (left < 0) left = 0;
    if (top < 0) top = y + 14;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }

  // ════════════════════════════════════════════
  // Fig 2: q(t) and R(t) over rollouts, small multiples
  // ════════════════════════════════════════════
  function drawSeparation(chart, data) {
    const canvas = need(chart, 'canvas');
    const tip = need(chart, '.chart-tip');
    const latentBox = need(chart.closest('figure'), '[data-role=latent]');
    const top = data.axes.top, bot = data.axes.bottom;
    [['top x', top.xscale], ['top y', top.yscale], ['bottom x', bot.xscale], ['bottom y', bot.yscale]].forEach(([w, s]) => logAxis(s, 'Fig 2 ' + w));
    if (top.xlim[0] !== bot.xlim[0] || top.xlim[1] !== bot.xlim[1]) throw new Error('Fig 2: top and bottom rows must share x limits');
    if (data.tau.row !== 'top' || data.tau.style !== 'dashed') throw new Error('Fig 2: unsupported tau line spec ' + JSON.stringify(data.tau));
    const t = data.t, models = data.models;
    let layout = null, hoverIdx = -1;
    models.forEach((m) => {
      if (!/^#[0-9a-fA-F]{6}$/.test(m.color)) throw new Error('Fig 2: colour must be #rrggbb, got ' + m.color + ' for ' + m.name);
    });

    function overflowLabel(label) {
      if (label !== '$\\uparrow$overflow') throw new Error('Fig 2: unsupported overflow label ' + label);
      return '↑ overflow';
    }

    function computeLayout() {
      const W = chart.clientWidth;
      const cols = W >= 900 ? 6 : W >= 520 ? 3 : 2;
      const left = 50, right = 8, gapX = 12;
      const pw = (W - left - right - gapX * (cols - 1)) / cols;
      const titleH = 24, ph = Math.round(Math.max(70, pw * 0.66)), gapY = 10, tickH = 20, blockGap = 22;
      const blockH = titleH + ph + gapY + ph + tickH;
      const rows = Math.ceil(models.length / cols);
      const H = rows * blockH + (rows - 1) * blockGap + 22;
      const panels = models.map((m, i) => {
        const c = i % cols, r = Math.floor(i / cols);
        const x0 = left + c * (pw + gapX), y0 = r * (blockH + blockGap);
        const sx = logScale(top.xlim[0], top.xlim[1], x0, x0 + pw);
        const topY0 = y0 + titleH, botY0 = topY0 + ph + gapY;
        return {
          m, c, x0, x1: x0 + pw, y0, topY0, topY1: topY0 + ph, botY0, botY1: botY0 + ph, sx,
          syTop: logScale(top.ylim[0], top.ylim[1], topY0 + ph, topY0),
          syBot: logScale(bot.ylim[0], bot.ylim[1], botY0 + ph, botY0),
        };
      });
      return { W, H, cols, panels };
    }

    function drawPanelAxes(ctx, p, spec, y0, y1, sy, showX) {
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      spec.yticks.forEach((v) => line(ctx, p.x0, sy(v), p.x1, sy(v)));
      spec.xticks.forEach((v) => line(ctx, p.sx(v), y0, p.sx(v), y1));
      ctx.strokeStyle = FRAME;
      ctx.strokeRect(p.x0 + 0.5, y0 + 0.5, p.x1 - p.x0 - 1, y1 - y0 - 1);
      ctx.fillStyle = MUTED;
      ctx.font = FONT;
      if (p.c === 0) {
        ctx.textBaseline = 'middle';
        spec.yticks.forEach((v, k) => tickText(ctx, spec.yticklabels[k], p.x0 - 5, sy(v), 'right'));
      }
      if (showX) {
        if (!spec.xticklabels) throw new Error('Fig 2: bottom row needs x tick labels');
        ctx.textBaseline = 'top';
        spec.xticks.forEach((v, k) => tickText(ctx, spec.xticklabels[k], p.sx(v), y1 + 4, 'center'));
      }
    }

    function render() {
      layout = computeLayout();
      const ctx = sizeCanvas(canvas, layout.W, layout.H);
      ctx.clearRect(0, 0, layout.W, layout.H);
      const showLatent = latentBox.checked;
      layout.panels.forEach((p) => {
        const m = p.m;
        ctx.font = TITLE;
        ctx.fillStyle = INK;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(m.name, (p.x0 + p.x1) / 2, p.y0 + 16);

        drawPanelAxes(ctx, p, top, p.topY0, p.topY1, p.syTop, false);
        drawPanelAxes(ctx, p, bot, p.botY0, p.botY1, p.syBot, true);

        // top row: tau, latent q (gray), readout q with IQR band
        ctx.save();
        ctx.beginPath(); ctx.rect(p.x0, p.topY0, p.x1 - p.x0, p.topY1 - p.topY0); ctx.clip();
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = REF;
        ctx.lineWidth = 1;
        line(ctx, p.x0, p.syTop(data.tau.value), p.x1, p.syTop(data.tau.value));
        ctx.setLineDash([]);
        if (showLatent) {
          ctx.strokeStyle = data.colors.latent;
          ctx.lineWidth = 1.4;
          polyline(ctx, t, m.q_latent.center, p.sx, p.syTop);
        }
        ctx.fillStyle = m.color + '33';
        band(ctx, t, m.q_readout.q25, m.q_readout.q75, p.sx, p.syTop);
        ctx.strokeStyle = m.color;
        ctx.lineWidth = 1.8;
        polyline(ctx, t, m.q_readout.center, p.sx, p.syTop);
        ctx.restore();

        // bottom row: R(t) with IQR band
        ctx.save();
        ctx.beginPath(); ctx.rect(p.x0, p.botY0, p.x1 - p.x0, p.botY1 - p.botY0); ctx.clip();
        ctx.fillStyle = m.color + '33';
        band(ctx, t, m.r_err.q25, m.r_err.q75, p.sx, p.syBot);
        ctx.strokeStyle = m.color;
        ctx.lineWidth = 1.8;
        polyline(ctx, t, m.r_err.center, p.sx, p.syBot);
        ctx.restore();

        if (m.mp_dash !== null) {
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = m.color;
          ctx.lineWidth = 1;
          line(ctx, p.sx(m.mp_dash), p.topY0, p.sx(m.mp_dash), p.topY1);
          line(ctx, p.sx(m.mp_dash), p.botY0, p.sx(m.mp_dash), p.botY1);
          ctx.setLineDash([]);
        }
        if (m.overflow) {
          const o = m.overflow;
          if (o.coords !== 'data' || o.arrow !== false) throw new Error('Fig 2: unsupported overflow annotation ' + JSON.stringify(o));
          ctx.font = FONT;
          ctx.fillStyle = m.color;
          ctx.textAlign = o.ha;
          ctx.textBaseline = 'middle';
          ctx.fillText(overflowLabel(o.label), p.sx(o.text_xy[0]), Math.max(p.botY0 + 8, p.syBot(o.text_xy[1])));
        }

        if (hoverIdx >= 0) {
          const x = p.sx(t[hoverIdx]);
          ctx.strokeStyle = INK;
          ctx.lineWidth = 1;
          line(ctx, x, p.topY0, x, p.topY1);
          line(ctx, x, p.botY0, x, p.botY1);
        }
      });

      // shared axis titles
      ctx.fillStyle = MUTED;
      ctx.font = FONT;
      const first = layout.panels[0];
      ctx.save();
      ctx.translate(11, (first.topY0 + first.topY1) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('q(t)  (log)', 0, 0);
      ctx.restore();
      ctx.save();
      ctx.translate(11, (first.botY0 + first.botY1) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('R(t)  (log)', 0, 0);
      ctx.restore();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('step t', (layout.panels[0].x0 + layout.W) / 2, layout.H - 2);
    }

    function nearestIndex(tv) {
      let lo = 0, hi = t.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (t[mid] < tv) lo = mid; else hi = mid; }
      return Math.abs(Math.log(t[lo]) - Math.log(tv)) <= Math.abs(Math.log(t[hi]) - Math.log(tv)) ? lo : hi;
    }

    function onPointer(e) {
      const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const p = layout.panels.find((q) => x >= q.x0 && x <= q.x1 && y >= q.topY0 && y <= q.botY1);
      if (!p) { clearHover(); return; }
      hoverIdx = nearestIndex(p.sx.inv(x));
      render();
      const m = p.m, i = hoverIdx;
      const qr = m.q_readout, re = m.r_err;
      placeTip(tip, chart, x, y,
        '<strong>' + m.name + '</strong> &middot; t = ' + niceT(t[i]) +
        '<br>q(t) ' + fmt(qr.center[i]) + ' <span class="muted">IQR ' + fmt(qr.q25[i]) + '–' + fmt(qr.q75[i]) + '</span>' +
        (latentBox.checked ? '<br><span class="muted">latent q(t) ' + fmt(m.q_latent.center[i]) + '</span>' : '') +
        '<br>R(t) ' + fmt(re.center[i]) +
        (m.mp_dash !== null ? '<br><span class="muted">mp = ' + m.mp + '</span>' : '<br><span class="muted">mp = 0 (✗)</span>'));
    }
    function clearHover() {
      if (hoverIdx < 0) return;
      hoverIdx = -1;
      tip.hidden = true;
      render();
    }

    canvas.addEventListener('pointermove', onPointer);
    canvas.addEventListener('pointerdown', onPointer);
    canvas.addEventListener('pointerleave', clearHover);
    latentBox.addEventListener('change', render);
    window.addEventListener('resize', render);
    document.fonts.ready.then(render);
    render();
  }

  // ════════════════════════════════════════════
  // Fig 4: q_t against failure-normalized time t / mp
  // ════════════════════════════════════════════
  function drawCollapse(chart, data) {
    const canvas = need(chart, 'canvas');
    const tip = need(chart, '.chart-tip');
    const toggles = need(chart.closest('figure'), '[data-role=families]');
    const ax = data.axes;
    logAxis(ax.xscale, 'Fig 4 x');
    logAxis(ax.yscale, 'Fig 4 y');
    if (data.fit_line !== null) throw new Error('Fig 4: drawing a fit line is not supported');
    const x = data.x, runs = data.runs, corr = data.correlation;
    const colorOf = {}, shown = {};
    data.families.forEach((f) => { colorOf[f.label] = f.color; shown[f.label] = true; });
    runs.forEach((r) => {
      if (!(r.family in colorOf)) throw new Error('Fig 4: run of unknown family ' + r.family);
      if (!(r.t_cross > 0)) throw new Error('Fig 4: every exported run must have a crossing, got T_cross = ' + r.t_cross + ' for ' + r.shard);
    });
    if (runs.length !== corr.n) throw new Error('Fig 4: ' + runs.length + ' runs exported but the correlation uses ' + corr.n);
    let geo = null, hover = -1;

    data.families.forEach((f) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'toggle is-on';
      b.innerHTML = '<i class="line" style="background:' + f.color + '"></i>' + f.label + ' <span class="muted">' + f.n_runs + ' runs</span>';
      b.setAttribute('aria-pressed', 'true');
      b.addEventListener('click', () => {
        shown[f.label] = !shown[f.label];
        b.classList.toggle('is-on', shown[f.label]);
        b.setAttribute('aria-pressed', String(shown[f.label]));
        hover = -1;
        tip.hidden = true;
        render();
      });
      toggles.appendChild(b);
    });

    function refStyle(ctx, style) {
      if (style === 'dashed') ctx.setLineDash([5, 4]);
      else if (style === 'dotted') ctx.setLineDash([1.5, 3]);
      else throw new Error('Fig 4: unsupported reference line style ' + style);
    }

    function render() {
      const W = chart.clientWidth;
      const H = Math.round(W / (window.innerWidth < NARROW_VIEWPORT ? 1.15 : 1.5));
      const L = 46, R = 12, T = 10, B = 40;
      const sx = logScale(ax.xlim[0], ax.xlim[1], L, W - R);
      const sy = logScale(ax.ylim[0], ax.ylim[1], H - B, T);
      geo = { W, H, L, R, T, B, sx, sy };
      const ctx = sizeCanvas(canvas, W, H);
      ctx.clearRect(0, 0, W, H);

      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ax.xticks.forEach((v) => line(ctx, sx(v), T, sx(v), H - B));
      ax.yticks.forEach((v) => line(ctx, L, sy(v), W - R, sy(v)));
      ctx.strokeStyle = FRAME;
      ctx.strokeRect(L + 0.5, T + 0.5, W - L - R - 1, H - T - B - 1);
      ctx.fillStyle = MUTED;
      ctx.font = FONT;
      ctx.textBaseline = 'middle';
      ax.yticks.forEach((v, k) => tickText(ctx, ax.yticklabels[k], L - 6, sy(v), 'right'));
      ctx.textBaseline = 'top';
      ax.xticks.forEach((v, k) => tickText(ctx, ax.xticklabels[k], sx(v), H - B + 6, 'center'));
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('t / mp', (L + W - R) / 2, H - 2);
      ctx.save();
      ctx.translate(12, (T + H - B) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle';
      const yTitle = [{ t: 'q' }, { t: 't', sub: true }, { t: '  (log)' }];
      drawRich(ctx, yTitle, -richWidth(ctx, yTitle) / 2, 0);
      ctx.restore();

      ctx.save();
      ctx.beginPath(); ctx.rect(L, T, W - L - R, H - T - B); ctx.clip();
      ctx.strokeStyle = REF;
      ctx.lineWidth = 1;
      data.reference_lines.forEach((ref) => {
        refStyle(ctx, ref.style);
        if (ref.orientation === 'horizontal') line(ctx, L, sy(ref.value), W - R, sy(ref.value));
        else if (ref.orientation === 'vertical') line(ctx, sx(ref.value), T, sx(ref.value), H - B);
        else throw new Error('Fig 4: unsupported reference orientation ' + ref.orientation);
      });
      ctx.setLineDash([]);

      ctx.lineWidth = 1;
      ctx.globalAlpha = hover >= 0 ? 0.12 : 0.3;
      runs.forEach((r) => {
        if (!shown[r.family]) return;
        ctx.strokeStyle = colorOf[r.family];
        polyline(ctx, x, r.q, sx, sy);
      });
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2.8;
      data.families.forEach((f) => {
        if (!shown[f.label]) return;
        ctx.strokeStyle = f.color;
        polyline(ctx, x, f.median, sx, sy);
      });
      if (hover >= 0) {
        const r = runs[hover];
        ctx.strokeStyle = INK;
        ctx.lineWidth = 2;
        polyline(ctx, x, r.q, sx, sy);
        ctx.fillStyle = INK;
        ctx.beginPath();
        ctx.arc(sx(r.t_cross / r.mp), sy(data.tcross_params.tau), 4, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.restore();

      // annotation box, bottom right, as in the paper figure
      const lines = [
        [{ t: 'Pearson r = ' + (corr.r >= 0 ? '+' : '−') + Math.abs(corr.r).toFixed(3) + ' (log-log)' }],
        pRuns(corr.p),
        [{ t: 'N = ' + corr.n + ' runs' }],
      ];
      ctx.font = FONT;
      const bw = Math.max(...lines.map((l) => richWidth(ctx, l))) + 18, bh = lines.length * 17 + 10;
      const bx = W - R - bw - 6, by = H - B - bh - 6;
      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = FRAME;
      ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
      ctx.fillStyle = INK;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      lines.forEach((l, k) => drawRich(ctx, l, bx + 9, by + 6 + k * 17));
    }

    function onPointer(e) {
      const r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      const { sx, sy, L, R, T, B, W, H } = geo;
      if (px < L || px > W - R || py < T || py > H - B) { clearHover(); return; }
      const xv = sx.inv(px);
      let lo = 0, hi = x.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (x[mid] < xv) lo = mid; else hi = mid; }
      const i = Math.abs(Math.log(x[lo] / xv)) <= Math.abs(Math.log(x[hi] / xv)) ? lo : hi;
      let best = -1, bestD = 14;
      runs.forEach((run, k) => {
        if (!shown[run.family]) return;
        const q = run.q[i];
        if (q === null || !(q > 0)) return;
        const d = Math.abs(sy(q) - py);
        if (d < bestD) { bestD = d; best = k; }
      });
      if (best !== hover) { hover = best; render(); }
      if (best < 0) { tip.hidden = true; return; }
      const run = runs[best];
      placeTip(tip, chart, px, py,
        '<strong>' + run.family + '</strong><br><span class="muted">d_state ' + run.d_state + ' · lr ' + run.lr + ' · ' + run.scheduler + ' · seed ' + run.seed + '</span>' +
        '<br>mp = ' + run.mp + ' · T_cross = ' + run.t_cross);
    }
    function clearHover() {
      tip.hidden = true;
      if (hover < 0) return;
      hover = -1;
      render();
    }

    canvas.addEventListener('pointermove', onPointer);
    canvas.addEventListener('pointerdown', onPointer);
    canvas.addEventListener('pointerleave', clearHover);
    window.addEventListener('resize', render);
    document.fonts.ready.then(render);
    render();
  }

  const DRAWERS = { separation: drawSeparation, collapse: drawCollapse };

  function init() {
    document.querySelectorAll('.chart[data-chart]').forEach((chart) => {
      const draw = DRAWERS[chart.dataset.chart];
      if (!draw) throw new Error('unknown chart type: ' + chart.dataset.chart);
      loadJSON(chart.dataset.src).then((data) => draw(chart, data)).catch((err) => fail(chart, err));
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
