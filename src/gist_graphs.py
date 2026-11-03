import base64
import copy
import html as html_module
import io
import json
import math
import pathlib
import warnings
from datetime import datetime
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


_TIME_SERIES_QUANTILES_TITLE = "Time Series Quantiles"
_TIME_SERIES_SPAGHETTI_TITLE = "Time Series Spaghetti"
_TIME_SERIES_QUANTILES_PER_WELL_TITLE = "Time Series Quantiles Per Well"
_TIME_SERIES_SPAGHETTI_PER_WELL_TITLE = "Time Series Spaghetti Per Well"

# Portal HTML limits: min-max downsampling preserves extrema; fewer lines/points still show the envelope.
_PER_WELL_SPAGHETTI_MAX_REALIZATIONS = 40
_PER_WELL_SPAGHETTI_MAX_GROUPS = 40
_PER_WELL_SPAGHETTI_MAX_POINTS_PER_GROUP = 80
_PER_WELL_QUANTILES_MAX_POINTS_PER_GROUP = 200

_RT_SELECTION_STYLES = {
    "Must Include": {"fill": "#329839", "stroke": "#000000"},
    "May Include": {"fill": "#FFF45A", "stroke": "#000000"},
    "Include in Forecast": {"fill": "#2CB7EE", "stroke": "#000000"},
    "Exclude": {"fill": "#AEAEAE", "stroke": "#000000"},
    "0bbl Disposal": {"fill": "#FFFFFF", "stroke": "#000000"},
}
_RT_CURVE_STYLES = {
    "Minimum": {"color": "#FF2B2B", "dash": "solid"},
    "Maximum": {"color": "#FFA22B", "dash": "solid"},
}
_PRESSURE_RANGES_CMAP = matplotlib.colormaps["viridis"]

# Total (all-well) time series HTML limits; min-max downsampling keeps each line's envelope.
_TOTAL_QUANTILES_MAX_POINTS_PER_GROUP = 400
_TOTAL_SPAGHETTI_MAX_REALIZATIONS = 100
_TOTAL_SPAGHETTI_MAX_POINTS_PER_GROUP = 200


# Shared zoom controls injected into every GIST HTML graph (portal iframe is sandboxed: no CDN).
# Drag = box zoom, Shift+drag = pan, double-click = reset, +/-/Reset buttons.
# There is deliberately no wheel handler so scrolling the results page never zooms a graph.
_ZOOM_CSS = """
  .gz-toolbar { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; }
  .gz-hint { color: #777; font-size: 11px; margin-right: 4px; }
  .gz-btn { font: inherit; font-size: 12px; min-width: 26px; height: 24px; padding: 0 8px; color: #222;
            border: 1px solid #bbb; border-radius: 4px; background: #f7f7f7; cursor: pointer; }
  .gz-btn:hover { background: #ececec; }
  .gz-btn:disabled { color: #aaa; cursor: default; background: #fafafa; }
"""

_ZOOM_JS = """
var GistZoom = (function () {
  var NS = 'http://www.w3.org/2000/svg';
  var MIN_FRACTION = 0.001;
  var DRAG_THRESHOLD = 5;

  function copy(d) { return { xmin: d.xmin, xmax: d.xmax, ymin: d.ymin, ymax: d.ymax }; }

  function svgPoint(svg, ev) {
    var ctm = svg.getScreenCTM();
    if (!ctm || !ctm.inverse) return null;
    var pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    return pt.matrixTransform(ctm.inverse());
  }

  function clampAxis(lo, hi, flo, fhi) {
    var fspan = fhi - flo;
    var minSpan = fspan * MIN_FRACTION;
    if (hi - lo < minSpan) {
      var c = (lo + hi) / 2;
      lo = c - minSpan / 2; hi = c + minSpan / 2;
    }
    if (hi - lo >= fspan) return [flo, fhi];
    if (lo < flo) { hi += flo - lo; lo = flo; }
    if (hi > fhi) { lo -= hi - fhi; hi = fhi; }
    return [lo, hi];
  }

  function attach(opts) {
    var svg = opts.svg, plot = opts.plot, axes = opts.axes || 'xy';
    var full = copy(opts.full), view = copy(opts.full);
    var listeners = [];

    function isZoomed() {
      return view.xmin !== full.xmin || view.xmax !== full.xmax || view.ymin !== full.ymin || view.ymax !== full.ymax;
    }
    function emit() {
      listeners.forEach(function (fn) { fn(); });
      opts.onChange(copy(view));
    }
    function setView(v) {
      var xs = clampAxis(v.xmin, v.xmax, full.xmin, full.xmax);
      view.xmin = xs[0]; view.xmax = xs[1];
      if (axes === 'xy') {
        var ys = clampAxis(v.ymin, v.ymax, full.ymin, full.ymax);
        view.ymin = ys[0]; view.ymax = ys[1];
      } else {
        view.ymin = full.ymin; view.ymax = full.ymax;
      }
      emit();
    }
    function toData(p) {
      return {
        x: view.xmin + (p.x - plot.x0) / (plot.x1 - plot.x0) * (view.xmax - view.xmin),
        y: view.ymax - (p.y - plot.y0) / (plot.y1 - plot.y0) * (view.ymax - view.ymin)
      };
    }
    var EDGE = 3; /* tolerance so a drag started right on the plot frame still counts */
    function inPlot(p) {
      return p && p.x >= plot.x0 - EDGE && p.x <= plot.x1 + EDGE && p.y >= plot.y0 - EDGE && p.y <= plot.y1 + EDGE;
    }
    function clampPt(p) {
      return { x: Math.max(plot.x0, Math.min(plot.x1, p.x)), y: Math.max(plot.y0, Math.min(plot.y1, p.y)) };
    }

    var start = null, mode = null, startView = null, box = null;

    function removeBox() {
      if (box && box.parentNode) box.parentNode.removeChild(box);
      box = null;
    }
    function onMove(ev) {
      var p = svgPoint(svg, ev);
      if (!p || !start) return;
      if (mode === 'pan') {
        var dx = (p.x - start.x) / (plot.x1 - plot.x0) * (startView.xmax - startView.xmin);
        var dy = (p.y - start.y) / (plot.y1 - plot.y0) * (startView.ymax - startView.ymin);
        setView({ xmin: startView.xmin - dx, xmax: startView.xmax - dx, ymin: startView.ymin + dy, ymax: startView.ymax + dy });
        return;
      }
      var q = clampPt(p);
      if (!box) {
        box = document.createElementNS(NS, 'rect');
        box.setAttribute('fill', 'rgba(23, 71, 158, 0.12)');
        box.setAttribute('stroke', '#17479E');
        box.setAttribute('stroke-dasharray', '4 3');
        box.setAttribute('pointer-events', 'none');
        svg.appendChild(box);
      }
      var bx = Math.min(start.x, q.x), bw = Math.abs(q.x - start.x);
      var by = axes === 'xy' ? Math.min(start.y, q.y) : plot.y0;
      var bh = axes === 'xy' ? Math.abs(q.y - start.y) : plot.y1 - plot.y0;
      box.setAttribute('x', bx); box.setAttribute('y', by);
      box.setAttribute('width', bw); box.setAttribute('height', bh);
    }
    function onUp(ev) {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      var wasBox = mode === 'box';
      removeBox();
      var p = svgPoint(svg, ev);
      var s = start;
      start = null; mode = null;
      if (!wasBox || !p || !s) return;
      var q = clampPt(p);
      var bigX = Math.abs(q.x - s.x) >= DRAG_THRESHOLD;
      var bigY = axes === 'xy' && Math.abs(q.y - s.y) >= DRAG_THRESHOLD;
      if (!bigX && !bigY) return;
      var a = toData(s), b = toData(q);
      var next = copy(view);
      if (bigX) { next.xmin = Math.min(a.x, b.x); next.xmax = Math.max(a.x, b.x); }
      if (bigY) { next.ymin = Math.min(a.y, b.y); next.ymax = Math.max(a.y, b.y); }
      setView(next);
    }

    svg.addEventListener('mousedown', function (ev) {
      if (ev.button !== 0) return;
      var p = svgPoint(svg, ev);
      if (!inPlot(p)) return;
      ev.preventDefault();
      start = clampPt(p);
      mode = ev.shiftKey ? 'pan' : 'box';
      startView = copy(view);
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    });
    svg.addEventListener('dblclick', function (ev) {
      if (inPlot(svgPoint(svg, ev))) ctrl.reset();
    });

    var ctrl = {
      reset: function () { view = copy(full); emit(); },
      setFull: function (d) { full = copy(d); view = copy(d); emit(); },
      zoomBy: function (factor) {
        var cx = (view.xmin + view.xmax) / 2, hx = (view.xmax - view.xmin) * factor / 2;
        var cy = (view.ymin + view.ymax) / 2, hy = (view.ymax - view.ymin) * factor / 2;
        setView({ xmin: cx - hx, xmax: cx + hx, ymin: cy - hy, ymax: cy + hy });
      },
      view: function () { return copy(view); },
      isZoomed: isZoomed,
      onState: function (fn) { listeners.push(fn); }
    };
    return ctrl;
  }

  function addButtons(host, ctrl) {
    var wrap = document.createElement('span');
    wrap.className = 'gz-toolbar';
    var hint = document.createElement('span');
    hint.className = 'gz-hint';
    hint.textContent = 'Drag to zoom \\u00b7 Shift+drag to pan';
    wrap.appendChild(hint);
    function button(label, title, fn) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'gz-btn'; b.textContent = label; b.title = title;
      b.setAttribute('aria-label', title);
      b.addEventListener('click', fn);
      wrap.appendChild(b);
      return b;
    }
    button('+', 'Zoom in', function () { ctrl.zoomBy(1 / 1.5); });
    var out = button('\\u2212', 'Zoom out', function () { ctrl.zoomBy(1.5); });
    var reset = button('Reset', 'Reset zoom (or double-click the graph)', function () { ctrl.reset(); });
    function sync() { out.disabled = !ctrl.isZoomed(); reset.disabled = !ctrl.isZoomed(); }
    ctrl.onState(sync);
    sync();
    host.appendChild(wrap);
  }

  function niceStep(span, count) {
    var raw = span / Math.max(1, count);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var n = raw / mag;
    var s = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
    return s * mag;
  }
  function ticks(min, max, count) {
    if (!(max > min) || !isFinite(min) || !isFinite(max)) return { values: [min], step: 1 };
    var step = niceStep(max - min, count);
    var values = [];
    var first = Math.ceil(min / step - 1e-9);
    for (var i = 0; i < 200; i++) {
      var v = (first + i) * step;
      if (v > max + step * 1e-9) break;
      values.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return { values: values, step: step };
  }
  function fmtTick(v, step) {
    var d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    var scaled = step * Math.pow(10, d);
    if (Math.abs(scaled - Math.round(scaled)) > 1e-6) d += 1;
    return v.toFixed(Math.min(d, 6));
  }

  return { attach: attach, addButtons: addButtons, ticks: ticks, fmtTick: fmtTick };
})();
"""

# Pressure Ranges zooms the PSI (x) axis only: rows are wells and already scroll vertically.
# The server-rendered markup is the unzoomed view; this script repositions it on zoom.
_PRESSURE_RANGES_ZOOM_SCRIPT = """
(function () {
  var cfg = JSON.parse(document.getElementById('pr-zoom-config').textContent);
  var NS = 'http://www.w3.org/2000/svg';
  var plotSvg = document.getElementById('prPlot');
  var gridG = document.getElementById('prGrid');
  var tickG = document.getElementById('prTicks');
  var W = cfg.plotWidth;
  var circles = Array.prototype.map.call(plotSvg.querySelectorAll('circle'), function (el) {
    return { el: el, psi: parseFloat(el.getAttribute('cx')) / W * cfg.xMax };
  });
  function makeEl(name, attrs) {
    var el = document.createElementNS(NS, name);
    Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  function rebuildTicks(values, step, sx) {
    clear(gridG); clear(tickG);
    values.forEach(function (v) {
      var label = step === null ? String(v) : GistZoom.fmtTick(v, step);
      var tx = sx(v).toFixed(1);
      gridG.appendChild(makeEl('line', {
        'class': 'gridline', 'data-tick': label, x1: tx, y1: 0, x2: tx, y2: cfg.gridY2,
        stroke: '#e5e5e5', 'stroke-width': 1
      }));
      var t = makeEl('text', { 'class': 'x-tick', 'data-tick': label, x: tx, y: 16, 'text-anchor': 'middle' });
      t.textContent = label;
      tickG.appendChild(t);
    });
  }
  function draw(v) {
    var span = v.xmax - v.xmin;
    function sx(p) { return (p - v.xmin) / span * W; }
    var zoomed = v.xmin > 0 || v.xmax < cfg.xMax;
    plotSvg.classList.toggle('pr-zoomed', zoomed);
    circles.forEach(function (c) { c.el.setAttribute('cx', sx(c.psi).toFixed(2)); });
    if (zoomed) {
      var t = GistZoom.ticks(v.xmin, v.xmax, 9);
      rebuildTicks(t.values, t.step, sx);
    } else {
      rebuildTicks(cfg.ticks, null, sx);
    }
  }
  var zoom = GistZoom.attach({
    svg: plotSvg,
    plot: { x0: 0, y0: 0, x1: W, y1: cfg.svgHeight },
    full: { xmin: 0, xmax: cfg.xMax, ymin: 0, ymax: 1 },
    axes: 'x',
    onChange: draw
  });
  GistZoom.addButtons(document.getElementById('prZoomBar'), zoom);
})();
"""


# Interactive R-t plot: marker area scales with MMBBL (Matplotlib-equivalent clip); hover tooltips.
_RT_PLOT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>R-t plot</title>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #fff; }
  #wrap { box-sizing: border-box; height: 100%; padding: 8px 10px 10px; display: flex; flex-direction: column; }
  #head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
  #legend { flex: 1 1 auto; display: flex; flex-wrap: wrap; gap: 8px 14px; justify-content: center; align-items: center; font-size: 11px; }
  #legend .item { display: inline-flex; align-items: center; gap: 4px; }
  #legend .sw { width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(0,0,0,.75); flex-shrink: 0; }
  #legend .line-swatch { width: 22px; height: 0; flex-shrink: 0; border-top-width: 2px; border-top-style: solid; display: inline-block; vertical-align: middle; }
  #zoomBar { flex: 0 0 auto; }
  #svgHost { flex: 1; min-height: 200px; position: relative; }
  svg { display: block; width: 100%; height: 100%; user-select: none; }
  #tip {
    position: absolute; display: none; pointer-events: none; z-index: 10;
    background: rgba(30, 30, 35, 0.94); color: #f5f5f5; padding: 8px 10px; border-radius: 6px;
    font-size: 12px; line-height: 1.45; max-width: 280px; box-shadow: 0 2px 10px rgba(0,0,0,.2);
  }
  #tip .k { color: #aaa; }
___ZOOM_CSS___
</style>
</head>
<body>
<div id="wrap">
  <div id="head">
    <div id="legend"></div>
    <div id="zoomBar"></div>
  </div>
  <div id="svgHost">
    <svg id="chart" viewBox="0 0 1000 560" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
      <rect id="hitLayer" x="0" y="0" width="1000" height="560" fill="transparent" cursor="crosshair"/>
    </svg>
    <div id="tip"></div>
  </div>
</div>
<script type="application/json" id="rt-payload">___PAYLOAD___</script>
<script>
___ZOOM_JS___
</script>
<script>
(function () {
  const payload = JSON.parse(document.getElementById('rt-payload').textContent);
  const W = 1000, H = 560;
  const ml = 72, mr = 28, mt = 8, mb = 52;
  const plotW = W - ml - mr, plotH = H - mt - mb;
  const x0 = ml, y0 = mt, x1 = ml + plotW, y1 = mt + plotH;

  const xLabel = payload.xLabel;
  const yLabel = payload.yLabel;
  function escapeHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  const pts = payload.points || [];
  const curves = payload.curves || [];
  let xmin = Infinity, xmax = -Infinity, ymin = 0, ymax = -Infinity;
  pts.forEach(function (p) {
    if (p.x < xmin) xmin = p.x;
    if (p.x > xmax) xmax = p.x;
    if (p.y < ymin) ymin = p.y;
    if (p.y > ymax) ymax = p.y;
  });
  curves.forEach(function (c) {
    (c.points || []).forEach(function (q) {
      var x = q[0], y = q[1];
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    });
  });
  if (!isFinite(xmin) || !isFinite(xmax) || xmin === xmax) { xmin = -1; xmax = 1; }
  if (!isFinite(ymax) || ymin === ymax) { ymin = 0; ymax = 1; }
  var xpad = (xmax - xmin) * 0.03 || 0.5;
  var ypad = (ymax - ymin) * 0.05 || 0.5;
  xmin -= xpad; xmax += xpad;
  ymin = Math.max(0, ymin - ypad);
  ymax += ypad;
  const full = { xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax };

  const svg = document.getElementById('chart');
  const hitLayer = document.getElementById('hitLayer');
  const NS = 'http://www.w3.org/2000/svg';
  function makeEl(name, attrs) {
    var el = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  /* Static layers: clip region, axis frame and axis titles. Grid, ticks and data redraw on zoom. */
  const defs = makeEl('defs');
  const clip = makeEl('clipPath', { id: 'rtPlotClip' });
  clip.appendChild(makeEl('rect', { x: x0, y: y0, width: plotW, height: plotH }));
  defs.appendChild(clip);
  svg.insertBefore(defs, hitLayer);
  const gridG = makeEl('g', { class: 'grid' });
  const dataG = makeEl('g', { 'clip-path': 'url(#rtPlotClip)' });
  const axisG = makeEl('g');
  const tickG = makeEl('g');
  [gridG, dataG, axisG, tickG].forEach(function (g) { svg.insertBefore(g, hitLayer); });

  axisG.appendChild(makeEl('rect', { x: x0, y: y0, width: plotW, height: plotH, fill: 'none', stroke: '#333', 'stroke-width': '1.2' }));
  var xl = makeEl('text', { x: (x0 + x1) / 2, y: H - 10, 'text-anchor': 'middle', fill: '#111', 'font-size': '13', 'font-weight': '600' });
  xl.textContent = xLabel;
  axisG.appendChild(xl);
  var yl = makeEl('text', {
    transform: 'rotate(-90 ' + (x0 - 46) + ' ' + ((y0 + y1) / 2) + ')',
    x: x0 - 46, y: (y0 + y1) / 2, 'text-anchor': 'middle', fill: '#111', 'font-size': '13', 'font-weight': '600'
  });
  yl.textContent = yLabel;
  axisG.appendChild(yl);

  const defaultR = (typeof payload.pointRadius === 'number' && isFinite(payload.pointRadius)) ? payload.pointRadius : 5;
  const plotPts = [];

  function draw(v) {
    clear(gridG); clear(dataG); clear(tickG);
    plotPts.length = 0;
    function xScale(x) { return x0 + (x - v.xmin) / (v.xmax - v.xmin) * plotW; }
    function yScale(y) { return y1 - (y - v.ymin) / (v.ymax - v.ymin) * plotH; }

    var xt = GistZoom.ticks(v.xmin, v.xmax, 7);
    xt.values.forEach(function (xv) {
      var gx = xScale(xv);
      gridG.appendChild(makeEl('line', { x1: gx, x2: gx, y1: y0, y2: y1, stroke: '#e8e8e8', 'stroke-width': '1' }));
      var t = makeEl('text', { x: gx, y: y1 + 18, 'text-anchor': 'middle', fill: '#333', 'font-size': '11' });
      t.textContent = GistZoom.fmtTick(xv, xt.step);
      tickG.appendChild(t);
    });
    var yt = GistZoom.ticks(v.ymin, v.ymax, 7);
    yt.values.forEach(function (yv) {
      var gy = yScale(yv);
      gridG.appendChild(makeEl('line', { x1: x0, x2: x1, y1: gy, y2: gy, stroke: '#e8e8e8', 'stroke-width': '1' }));
      var t = makeEl('text', { x: x0 - 8, y: gy + 4, 'text-anchor': 'end', fill: '#333', 'font-size': '11' });
      t.textContent = GistZoom.fmtTick(yv, yt.step);
      tickG.appendChild(t);
    });

    /* Circles first, then diffusivity curves, so lines stay in the foreground. */
    pts.forEach(function (p) {
      var cx = xScale(p.x), cy = yScale(p.y);
      var pr = (typeof p.r === 'number' && isFinite(p.r) && p.r > 0) ? p.r : defaultR;
      if (cx < x0 - pr || cx > x1 + pr || cy < y0 - pr || cy > y1 + pr) return;
      plotPts.push({ cx: cx, cy: cy, r: pr, p: p });
      dataG.appendChild(makeEl('circle', {
        cx: cx, cy: cy, r: String(pr), fill: p.color || '#333', stroke: p.stroke || '#000',
        'stroke-width': '1', opacity: '0.88', 'pointer-events': 'none'
      }));
    });

    curves.forEach(function (curve) {
      var d = '';
      (curve.points || []).forEach(function (pt, i) {
        d += (i ? ' L ' : 'M ') + xScale(pt[0]).toFixed(2) + ' ' + yScale(pt[1]).toFixed(2) + ' ';
      });
      var path = makeEl('path', { d: d.trim(), fill: 'none', stroke: curve.color || '#222', 'stroke-width': '2' });
      if (curve.dash === 'dash') path.setAttribute('stroke-dasharray', '7 5');
      dataG.appendChild(path);
    });
  }

  const leg = document.getElementById('legend');
  (payload.legendItems || []).forEach(function (item) {
    var div = document.createElement('div');
    div.className = 'item';
    if (item.type === 'scatter') {
      div.innerHTML = '<span class="sw" style="background:' + item.color + '"></span><span>' + escapeHtml(item.label) + '</span>';
    } else if (item.type === 'note') {
      div.innerHTML = '<span style="font-style:italic;color:#555">' + escapeHtml(item.label) + '</span>';
    } else {
      var st = item.dash ? 'dashed' : 'solid';
      div.innerHTML = '<span class="line-swatch" style="border-top-color:' + item.color + ';border-top-style:' + st + '"></span><span>' + escapeHtml(item.label) + '</span>';
    }
    leg.appendChild(div);
  });

  const tip = document.getElementById('tip');
  const host = document.getElementById('svgHost');

  function dist2(ax, ay, bx, by) { var dx = ax - bx, dy = ay - by; return dx * dx + dy * dy; }

  function showTip(clientX, clientY, p) {
    var uicShow = (p.uic !== undefined && p.uic !== null && String(p.uic).length) ? escapeHtml(String(p.uic)) : 'N/A';
    var nameLine = '';
    if (p.wellName !== undefined && p.wellName !== null && String(p.wellName).length) {
      nameLine = '<div><span class="k">Well Name</span> ' + escapeHtml(String(p.wellName)) + '</div>';
    }
    var xv = (typeof p.x === 'number' && isFinite(p.x)) ? p.x.toFixed(3) : String(p.x);
    var yv = (typeof p.y === 'number' && isFinite(p.y)) ? p.y.toFixed(2) : String(p.y);
    var mmbblLine = '';
    if (typeof p.mmbbl === 'number' && isFinite(p.mmbbl)) {
      mmbblLine = '<div><span class="k">MMBBL</span> ' + p.mmbbl.toFixed(4) + '</div>';
    }
    tip.innerHTML =
      '<div><span class="k">UIC Number</span> ' + uicShow + '</div>' +
      nameLine +
      mmbblLine +
      '<div><span class="k">' + escapeHtml(xLabel) + '</span> ' + xv + '</div>' +
      '<div><span class="k">' + escapeHtml(yLabel) + '</span> ' + yv + '</div>';
    tip.style.display = 'block';
    var rect = host.getBoundingClientRect();
    var tx = clientX - rect.left + 12, ty = clientY - rect.top + 12;
    tip.style.left = Math.min(tx, rect.width - 200) + 'px';
    tip.style.top = Math.min(ty, rect.height - 100) + 'px';
  }

  function hideTip() { tip.style.display = 'none'; }

  svg.addEventListener('mousemove', function (ev) {
    if (ev.buttons & 1) { hideTip(); return; }
    var pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    var ctm = svg.getScreenCTM();
    if (!ctm || !ctm.inverse) return;
    var loc = pt.matrixTransform(ctm.inverse());
    var lx = loc.x, ly = loc.y;
    if (lx < x0 || lx > x1 || ly < y0 || ly > y1) { hideTip(); return; }
    var best = null, bestD = Infinity;
    plotPts.forEach(function (o) {
      var d = dist2(lx, ly, o.cx, o.cy);
      var thr = (o.r + 14) * (o.r + 14);
      if (d < thr && d < bestD) { bestD = d; best = o; }
    });
    if (best) showTip(ev.clientX, ev.clientY, best.p);
    else hideTip();
  });
  svg.addEventListener('mouseleave', hideTip);
  svg.addEventListener('mousedown', hideTip);

  hitLayer.setAttribute('pointer-events', 'none');

  const zoom = GistZoom.attach({
    svg: svg,
    plot: { x0: x0, y0: y0, x1: x1, y1: y1 },
    full: full,
    axes: 'xy',
    onChange: draw
  });
  GistZoom.addButtons(document.getElementById('zoomBar'), zoom);
  draw(zoom.view());
})();
</script>
</body>
</html>
"""


def _with_zoom(template):
    """Fill the shared zoom CSS/JS placeholders of an HTML graph template."""
    return template.replace("___ZOOM_CSS___", _ZOOM_CSS).replace("___ZOOM_JS___", _ZOOM_JS)


def _json_for_script(payload):
    """Serialize a payload for an inline <script> block (a '</' in data must not close the tag)."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def _graphs_dir(helper):
    path = pathlib.Path(helper.scratchPath) / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_time_series_df(df, required_columns):
    if df is None or df.empty or not set(required_columns).issubset(df.columns):
        return pd.DataFrame()

    plot_df = df.copy()
    if "timestamp" in plot_df.columns:
        plot_df["timestamp"] = pd.to_numeric(plot_df["timestamp"], errors="coerce")
    else:
        plot_df["timestamp"] = pd.to_datetime(plot_df["Date"], errors="coerce").dt.as_unit("ns").to_numpy().view("int64") // 10**6
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    plot_df["DeltaPressure"] = pd.to_numeric(plot_df["DeltaPressure"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Date", "DeltaPressure", "timestamp"])
    if plot_df.empty:
        return pd.DataFrame()
    plot_df["timestamp"] = plot_df["timestamp"].astype("int64")
    return plot_df.sort_values("timestamp").reset_index(drop=True)


def _date_to_ms(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return int(ts.value // 10**6)


def _safe_float(value):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return None


def _build_time_series_norm(values, default_min=0.0, default_max=1.0):
    finite_values = []
    for value in values:
        numeric_value = _safe_float(value)
        if numeric_value is not None:
            finite_values.append(numeric_value)
    if not finite_values:
        return mcolors.Normalize(vmin=default_min, vmax=default_max)
    min_value = min(finite_values)
    max_value = max(finite_values)
    if math.isclose(min_value, max_value):
        max_value = min_value + 1.0
    return mcolors.Normalize(vmin=min_value, vmax=max_value)


def _rounded_float(value, digits):
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return None
    return round(numeric_value, digits)


def _downsample_time_series_points(points, max_points):
    if max_points is None or len(points) <= max_points:
        return points
    if max_points <= 2:
        return [points[0], points[-1]]

    interior = points[1:-1]
    if not interior:
        return [points[0], points[-1]]

    target_interior = max_points - 2
    bucket_size = max(1, math.ceil(len(interior) / target_interior))
    sampled_points = [points[0]]

    for start_idx in range(0, len(interior), bucket_size):
        bucket = interior[start_idx:start_idx + bucket_size]
        if not bucket:
            continue
        min_point = min(bucket, key=lambda point: point[1])
        max_point = max(bucket, key=lambda point: point[1])
        if min_point[0] <= max_point[0]:
            sampled_points.extend([min_point, max_point])
        else:
            sampled_points.extend([max_point, min_point])

    sampled_points.append(points[-1])
    sampled_points.sort(key=lambda point: point[0])

    deduped_points = []
    last_point = None
    for point in sampled_points:
        if last_point is None or point != last_point:
            deduped_points.append(point)
            last_point = point

    if len(deduped_points) <= max_points:
        return deduped_points

    step = max(1, math.ceil((len(deduped_points) - 2) / (max_points - 2)))
    trimmed_points = [deduped_points[0]]
    trimmed_points.extend(deduped_points[1:-1:step][:max_points - 2])
    trimmed_points.append(deduped_points[-1])
    return trimmed_points


def _build_time_series_payload(df, group_column, color_column=None, max_groups=None, max_points_per_group=None):
    payload_df = _coerce_time_series_df(df, ["Date", "DeltaPressure", group_column])
    if payload_df.empty:
        return []

    grouped = payload_df.groupby(group_column, sort=True)
    if max_groups is not None and grouped.ngroups > max_groups:
        # Subsample groups by sorted key without materializing every sub-frame.
        keys = list(grouped.groups.keys())
        step = max(1, math.ceil(len(keys) / max_groups))
        keep = set(keys[::step][:max_groups])
        groups = [(k, g) for k, g in grouped if k in keep]
    else:
        # Iterate the groupby directly; avoids building a list of all sub-frames.
        groups = grouped

    series = []
    for group_value, group_df in groups:
        timestamps = pd.to_numeric(group_df["timestamp"], errors="coerce").to_numpy(dtype="int64")
        pressures = pd.to_numeric(group_df["DeltaPressure"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(timestamps) & np.isfinite(pressures)
        if not np.any(valid):
            continue
        # Vectorized mask + bulk round, then zip native lists. Avoids
        # per-element numpy indexing over the full (often dense) series.
        valid_idx = np.flatnonzero(valid)
        ts_list = timestamps[valid_idx].tolist()
        pr_list = np.round(pressures[valid_idx], 3).tolist()
        points = [[int(t), float(p)] for t, p in zip(ts_list, pr_list)]
        if not points:
            continue
        points = _downsample_time_series_points(points, max_points_per_group)

        color_value = None
        if color_column and color_column in group_df.columns:
            first_value = group_df[color_column].dropna()
            if not first_value.empty:
                color_value = _safe_float(first_value.iloc[0])
        if color_value is None:
            color_value = _safe_float(group_value)

        series_payload = {
            "key": str(group_value),
            "colorValue": _rounded_float(color_value, 6),
            "points": points,
        }
        if group_column != "Realization":
            series_payload["label"] = _format_numeric_identifier_for_tooltip(group_value)
        series.append(series_payload)
    return series


def _split_interval_volume_across_months(subgraph, interval_end, interval_start, volume_bbl, interval_days):
    """
    Allocate interval volume and coverage days across calendar months when the window crosses a boundary.

    Each row represents a backward-averaged rate over ``interval_days`` ending at
    ``interval_end``. Volume and days are split in proportion to calendar-day overlap per month.
    """
    allocations = []
    start_period = interval_start.to_period("M")
    end_period = interval_end.to_period("M")
    total_days = float(interval_days)
    if total_days <= 0:
        return allocations

    for period in pd.period_range(start_period, end_period, freq="M"):
        month_start = period.to_timestamp()
        month_end_exclusive = (period + 1).to_timestamp()
        overlap_start = max(interval_start, month_start)
        overlap_end = min(interval_end, month_end_exclusive)
        if overlap_end <= overlap_start:
            continue
        overlap_days = (overlap_end - overlap_start).total_seconds() / 86400.0
        if overlap_days <= 0:
            continue
        allocated_volume = volume_bbl * (overlap_days / total_days)
        allocations.append(
            {
                "subgraph": subgraph,
                "month": period,
                "volume_bbl": allocated_volume,
                "coverage_days": overlap_days,
            }
        )
    return allocations


def _aggregate_disposal_to_monthly_bpd(disposal_df):
    """
    Roll up interval-averaged disposal (BPD) to monthly average daily rates for per-well graphs.

    Each input row is a backward-averaged daily rate (bbl/day) over the preceding interval
    ending at Date (from injectionV3.regularize). For each well/month the plotted BPD is
    total represented barrels divided by total represented calendar-day coverage.
    When an interval crosses a month boundary, volume and days are split proportionally by
    calendar-day overlap.
    """
    if disposal_df is None or disposal_df.empty:
        return disposal_df if disposal_df is not None else pd.DataFrame()
    if not {"Date", "BPD", "subgraph"}.issubset(disposal_df.columns):
        return pd.DataFrame(columns=list(disposal_df.columns))

    plot_df = disposal_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    plot_df["BPD"] = pd.to_numeric(plot_df["BPD"], errors="coerce")
    if "Days" in plot_df.columns:
        plot_df["Days"] = pd.to_numeric(plot_df["Days"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Date", "BPD", "subgraph"])
    if plot_df.empty:
        return plot_df

    sort_cols = ["subgraph", "Days", "Date"] if "Days" in plot_df.columns else ["subgraph", "Date"]
    plot_df = plot_df.sort_values(sort_cols).reset_index(drop=True)

    grouped = plot_df.groupby("subgraph", sort=False)
    if "Days" in plot_df.columns:
        plot_df["interval_days"] = grouped["Days"].diff()
    else:
        plot_df["interval_days"] = grouped["Date"].diff().dt.days

    median_interval = grouped["interval_days"].transform(
        lambda values: values.dropna().median() if values.dropna().size else 1.0
    )
    plot_df["interval_days"] = plot_df["interval_days"].fillna(median_interval)
    plot_df["interval_days"] = plot_df["interval_days"].clip(lower=1.0)
    plot_df["volume_bbl"] = plot_df["BPD"] * plot_df["interval_days"]
    plot_df["interval_start"] = plot_df["Date"] - pd.to_timedelta(plot_df["interval_days"], unit="D")

    plot_df["start_month"] = plot_df["interval_start"].dt.to_period("M")
    plot_df["end_month"] = plot_df["Date"].dt.to_period("M")
    crosses_boundary = plot_df["start_month"] != plot_df["end_month"]

    monthly_chunks = []
    same_month = plot_df.loc[~crosses_boundary, ["subgraph", "end_month", "volume_bbl", "interval_days"]].rename(
        columns={"end_month": "month", "interval_days": "coverage_days"}
    )
    if not same_month.empty:
        monthly_chunks.append(same_month)

    if crosses_boundary.any():
        boundary_rows = []
        for row in plot_df.loc[crosses_boundary].itertuples(index=False):
            boundary_rows.extend(
                _split_interval_volume_across_months(
                    row.subgraph,
                    row.Date,
                    row.interval_start,
                    row.volume_bbl,
                    row.interval_days,
                )
            )
        if boundary_rows:
            monthly_chunks.append(pd.DataFrame(boundary_rows))

    if not monthly_chunks:
        return pd.DataFrame(columns=list(disposal_df.columns))

    monthly = pd.concat(monthly_chunks, ignore_index=True)
    monthly = monthly.groupby(["subgraph", "month"], as_index=False).agg(
        volume_bbl=("volume_bbl", "sum"),
        coverage_days=("coverage_days", "sum"),
    )
    monthly["BPD"] = monthly["volume_bbl"] / monthly["coverage_days"]
    monthly = monthly.drop(columns=["volume_bbl", "coverage_days"])
    monthly["Date"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.drop(columns=["month"]).sort_values(["subgraph", "Date"]).reset_index(drop=True)
    return monthly


# The per-well quantiles and spaghetti graphs are built from the same disposal
# dataframe; remember the last payload so the (slow) monthly roll-up runs once.
# The cache is keyed on a content hash, so an edited or different frame is rebuilt.
_DISPOSAL_PAYLOAD_CACHE = {"key": None, "payload": None}


def _disposal_payload_cache_key(disposal_df):
    try:
        content_hash = pd.util.hash_pandas_object(disposal_df, index=True).to_numpy()
    except (TypeError, ValueError):
        return None
    return (tuple(disposal_df.columns), content_hash.tobytes())


def _build_disposal_payload(disposal_df):
    if disposal_df is None or disposal_df.empty:
        return {}
    cache_key = _disposal_payload_cache_key(disposal_df)
    if cache_key is not None and _DISPOSAL_PAYLOAD_CACHE["key"] == cache_key:
        return copy.deepcopy(_DISPOSAL_PAYLOAD_CACHE["payload"])
    payload = _build_disposal_payload_uncached(disposal_df)
    if cache_key is not None:
        _DISPOSAL_PAYLOAD_CACHE["key"] = cache_key
        _DISPOSAL_PAYLOAD_CACHE["payload"] = copy.deepcopy(payload)
    return payload


def _build_disposal_payload_uncached(disposal_df):
    if disposal_df is None or disposal_df.empty:
        return {}
    if not {"Date", "BPD", "subgraph"}.issubset(disposal_df.columns):
        return {}

    plot_df = _aggregate_disposal_to_monthly_bpd(disposal_df)
    if plot_df.empty:
        return {}
    if "timestamp" in plot_df.columns:
        plot_df["timestamp"] = pd.to_numeric(plot_df["timestamp"], errors="coerce")
    else:
        plot_df["timestamp"] = pd.to_datetime(plot_df["Date"], errors="coerce").dt.as_unit("ns").to_numpy().view("int64") // 10**6
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    plot_df["BPD"] = pd.to_numeric(plot_df["BPD"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Date", "BPD", "subgraph", "timestamp"]).sort_values("timestamp")
    if plot_df.empty:
        return {}

    disposal_by_well = {}
    for well_name, well_df in plot_df.groupby("subgraph", sort=True):
        # timestamp and BPD are already coerced to numeric and dropna'd above,
        # so every row is valid here. Bulk-round BPD and zip native lists
        # instead of itertuples + getattr per row.
        ts_list = well_df["timestamp"].to_numpy().tolist()
        bpd_list = np.round(well_df["BPD"].to_numpy(dtype=float), 2).tolist()
        points = [[x, y] for x, y in zip(ts_list, bpd_list)]
        if points:
            disposal_by_well[str(well_name)] = points
    return disposal_by_well


def _format_numeric_identifier_for_tooltip(v):
    """Format a scalar well identifier for display (integers without .0)."""
    try:
        fv = float(v)
        if fv == int(fv):
            return str(int(fv))
    except (TypeError, ValueError):
        pass
    return str(v)


def _uic_number_for_rt_tooltip(row):
    """Value for the R-t plot tooltip: UICNumber when present, else ID, else well name."""
    if "UICNumber" in row.index and pd.notna(row["UICNumber"]):
        return _format_numeric_identifier_for_tooltip(row["UICNumber"])
    if "ID" in row.index and pd.notna(row["ID"]):
        return _format_numeric_identifier_for_tooltip(row["ID"])
    if "Name" in row.index and pd.notna(row["Name"]):
        return str(row["Name"])
    return ""


def _well_name_for_rt_tooltip(row):
    """Well name for the R-t plot tooltip; empty string when the well has no name on file."""
    value = row["WellName"]
    if pd.isna(value):
        return ""
    return str(value).strip()


def _rt_marker_area_from_mmbbl(mmbbl):
    """Matplotlib scatter `s` equivalent: marker area = clip(MMBBL * 18, 30, 240)."""
    try:
        v = float(mmbbl)
    except (TypeError, ValueError):
        v = 0.0
    if not math.isfinite(v):
        v = 0.0
    return max(30.0, min(240.0, v * 18.0))


def _rt_svg_radius_from_mmbbl(mmbbl):
    """SVG circle radius in viewBox units from Matplotlib-equivalent area."""
    return math.sqrt(_rt_marker_area_from_mmbbl(mmbbl) / math.pi)


def filter_rt_plot_wells_future_start_date(
    well_df: pd.DataFrame,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Drop R-t plot wells whose StartDate is strictly after reference_date.

    Aligns with injection_updater_v5 future-start exclusion (calendar day).
    Wells activating on reference_date are kept; unparseable StartDate values are kept.
    """
    if well_df.empty or "StartDate" not in well_df.columns:
        return well_df

    ref = reference_date or datetime.now()
    ref_day = pd.Timestamp(ref.date())
    parsed = pd.to_datetime(well_df["StartDate"], format="mixed", errors="coerce")
    future_mask = parsed.notna() & (parsed.dt.normalize() > ref_day)

    before = len(well_df)
    filtered = well_df[~future_mask].copy()
    if len(filtered) < before:
        warnings.warn(
            f"R-t plot excluded {before - len(filtered)}/{before} wells "
            f"with StartDate after {ref_day.strftime('%Y-%m-%d')}.",
            UserWarning,
            stacklevel=2,
        )
    return filtered


def save_rt_plot_graph_artifact(helper, small_pp_df, well_df, artifact_key, display_order, title="R-t plot"):
    """Write the R-t plot as an interactive HTML graph artifact."""
    if small_pp_df.empty or well_df.empty:
        return

    artifact_path = _graphs_dir(helper) / f"{artifact_key}.html"

    plot_df = well_df.dropna(subset=["YearsInjectingToEarthquake", "Distances"]).copy()
    if plot_df.empty:
        return

    plot_df["Selection"] = plot_df["Selection"].fillna("Uncategorized")
    fallback_style = {"fill": "#C8C8C8", "stroke": "#000000"}

    x_label = "Years injecting before earthquake"
    y_label = "Distance from earthquake (km)"

    mmbbl_series = plot_df["MMBBL"] if "MMBBL" in plot_df.columns else pd.Series(0.0, index=plot_df.index)
    mm_numeric = pd.to_numeric(mmbbl_series, errors="coerce").fillna(0.0)

    points = []
    for i in range(len(plot_df)):
        row = plot_df.iloc[i]
        sel = row["Selection"]
        mmb_f = float(mm_numeric.iloc[i])
        points.append(
            {
                "uic": _uic_number_for_rt_tooltip(row),
                "wellName": _well_name_for_rt_tooltip(row),
                "x": round(float(row["YearsInjectingToEarthquake"]), 4),
                "y": round(float(row["Distances"]), 4),
                "mmbbl": round(mmb_f, 4),
                "r": round(_rt_svg_radius_from_mmbbl(mmb_f), 2),
                "color": _RT_SELECTION_STYLES.get(sel, fallback_style)["fill"],
                "stroke": _RT_SELECTION_STYLES.get(sel, fallback_style)["stroke"],
            }
        )

    curves = []
    for diffusivity, curve_df in small_pp_df.groupby("Diffusivity", sort=False):
        curve_df = curve_df.sort_values("Years Before Earthquake")
        curve_style = _RT_CURVE_STYLES.get(diffusivity, {"color": "#444444", "dash": "solid"})
        xy_pairs = np.round(curve_df[["Years Before Earthquake", "Distance"]].to_numpy(dtype=float), 4).tolist()
        curves.append(
            {
                "label": f"{diffusivity} diffusivity",
                "color": curve_style["color"],
                "dash": curve_style["dash"],
                "points": xy_pairs,
            }
        )

    legend_items = []
    seen_sel = set()
    for sel in plot_df["Selection"]:
        if sel in seen_sel:
            continue
        seen_sel.add(sel)
        legend_items.append(
            {
                "type": "scatter",
                "label": sel,
                "color": _RT_SELECTION_STYLES.get(sel, fallback_style)["fill"],
            }
        )
    legend_items.append(
        {
            "type": "note",
            "label": "Marker area scales with cumulative injection volume (MMBBL).",
        }
    )
    for c in curves:
        legend_items.append(
            {
                "type": "curve",
                "label": c["label"],
                "color": c["color"],
                "dash": c["dash"] == "dash",
            }
        )

    payload = {
        "title": title,
        "xLabel": x_label,
        "yLabel": y_label,
        "points": points,
        "curves": curves,
        "legendItems": legend_items,
    }
    html_out = _with_zoom(_RT_PLOT_HTML_TEMPLATE).replace("___PAYLOAD___", _json_for_script(payload))
    artifact_path.write_text(html_out, encoding="utf-8")

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption=(
            "Interactive R-t plot: marker size reflects MMBBL; hover for UIC number, well name, MMBBL, "
            "and axis values. Drag to zoom, Shift+drag to pan, double-click to reset."
        ),
        renderer="html",
        path=str(artifact_path),
        contentType="text/html",
        displayOrder=display_order,
        preferredHeight=600,
    )


def save_pressure_ranges_graph_artifact(
    helper,
    disaggregation_df,
    artifact_key,
    display_order,
    title="Pressure Ranges",
):
    """Write the Pressure Ranges plot as a scrollable HTML graph artifact."""
    required_columns = {"Pressures", "WellNo", "Order", "Name"}
    if disaggregation_df.empty or not required_columns.issubset(disaggregation_df.columns):
        return

    plot_df = disaggregation_df.dropna(subset=["Pressures", "WellNo", "Order", "Name"]).copy()
    if plot_df.empty:
        return

    ordered_wells = (
        plot_df[["Name", "WellNo"]]
        .groupby("Name", as_index=False)["WellNo"]
        .median()
        .sort_values("WellNo", ascending=False)
    )
    y_lookup = {name: idx for idx, name in enumerate(ordered_wells["Name"])}
    plot_df["Y"] = plot_df["Name"].map(y_lookup)

    well_count = len(ordered_wells)
    x_values = pd.to_numeric(plot_df["Pressures"], errors="coerce")
    order_values = pd.to_numeric(plot_df["Order"], errors="coerce")
    max_pressure = float(x_values.max()) if not x_values.empty else 1.0
    min_order = float(order_values.min()) if not order_values.empty else 1.0
    max_order = float(order_values.max()) if not order_values.empty else 1.0
    if not math.isfinite(max_pressure) or max_pressure <= 0:
        max_pressure = 1.0
    if not math.isfinite(min_order):
        min_order = 1.0
    if not math.isfinite(max_order) or max_order <= min_order:
        max_order = min_order + 1.0

    x_max = max_pressure * 1.04
    tick_step = 50 if x_max <= 350 else 100
    ticks = list(range(0, int(math.ceil(x_max / tick_step) * tick_step) + 1, tick_step))
    if ticks[-1] < x_max:
        ticks.append(int(math.ceil(x_max)))
    x_max = max(float(ticks[-1]), x_max)

    # Shared 3-column grid (names | plot | pad) so axis ticks and gridlines
    # stay in the same coordinate system at any iframe width, including fullscreen.
    label_width = 240
    plot_width = 900
    row_height = 30
    axis_height = 48
    svg_height = max(90, well_count * row_height + 16)

    def plot_x(value):
        """Map PSI to the plot-only SVG viewBox (0 .. plot_width)."""
        return (float(value) / x_max) * plot_width

    cmap = _PRESSURE_RANGES_CMAP

    def color_for_order(order):
        norm = (float(order) - min_order) / (max_order - min_order)
        norm = max(0.0, min(1.0, norm))
        rgba = cmap(norm)
        return "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
        )

    grid_lines = []
    x_tick_labels = []
    for tick in ticks:
        tx = plot_x(tick)
        grid_lines.append(
            f'<line class="gridline" data-tick="{html_module.escape(str(tick))}" '
            f'x1="{tx:.1f}" y1="0" x2="{tx:.1f}" y2="{svg_height - 16}" '
            f'stroke="#e5e5e5" stroke-width="1" />'
        )
        x_tick_labels.append(
            f'<text class="x-tick" data-tick="{html_module.escape(str(tick))}" '
            f'x="{tx:.1f}" y="16" text-anchor="middle">'
            f"{html_module.escape(str(tick))}</text>"
        )

    y_labels = []
    for name, y_index in y_lookup.items():
        y = y_index * row_height + row_height / 2
        y_labels.append(
            f'<text x="{label_width - 8}" y="{y + 4:.1f}" text-anchor="end">'
            f"{html_module.escape(str(name))}</text>"
        )

    points = []
    for _, row in plot_df.iterrows():
        x = plot_x(row["Pressures"])
        y = float(row["Y"]) * row_height + row_height / 2
        # cx carries 2 decimals: the zoom script recovers each point's PSI from it.
        points.append(
            f'<circle cx="{x:.2f}" cy="{y:.1f}" r="4.3" fill="{color_for_order(row["Order"])}" opacity="0.82" />'
        )

    zoom_config = {
        "plotWidth": plot_width,
        "svgHeight": svg_height,
        "gridY2": svg_height - 16,
        "xMax": x_max,
        "ticks": ticks,
    }

    legend_stops = []
    for pct in range(0, 101, 5):
        value = min_order + (max_order - min_order) * (pct / 100.0)
        legend_stops.append(f"{color_for_order(value)} {pct}%")

    legend_ticks = []
    order_tick_count = 5
    for i in range(order_tick_count):
        value = min_order + (max_order - min_order) * (i / (order_tick_count - 1))
        legend_ticks.append(
            f'<span class="legend-tick" style="left:{i * 25}%">{value:g}</span>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  html, body {{
    margin: 0;
    height: 100%;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: #fff;
    overflow: hidden;
  }}
  body {{
    box-sizing: border-box;
    padding: 12px 14px 10px;
    display: flex;
    flex-direction: column;
  }}
  .chart-grid {{
    display: grid;
    grid-template-columns: minmax(240px, 240fr) minmax(0, 900fr) minmax(28px, 32fr);
    width: 100%;
    box-sizing: border-box;
  }}
  .plot-scroll {{
    flex: 1 1 auto;
    min-height: 220px;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-gutter: stable;
    border-bottom: 1px solid #888;
  }}
  .plot-grid {{
    height: {svg_height}px;
  }}
  .axis-row {{
    flex: 0 0 {axis_height}px;
    height: {axis_height}px;
    overflow-y: hidden;
    overflow-x: hidden;
    scrollbar-gutter: stable;
  }}
  .chart-grid svg {{
    display: block;
    width: 100%;
    height: 100%;
    overflow: visible;
  }}
  .chart-grid svg.pr-zoomed {{
    overflow: hidden;
  }}
  #prPlot {{
    user-select: none;
  }}
  .pr-toolbar {{
    flex: 0 0 auto;
    display: flex;
    justify-content: flex-end;
    margin-bottom: 6px;
  }}
{_ZOOM_CSS}
  text {{
    fill: #1f1f1f;
    font-size: 12px;
  }}
  .x-label {{
    font-size: 13px;
  }}
  .legend-wrap {{
    flex: 0 0 72px;
    width: min(620px, 58%);
    margin: 0 auto;
    position: relative;
  }}
  .legend-bar {{
    height: 24px;
    margin-top: 14px;
    border: 1px solid #555;
    background: linear-gradient(to right, {", ".join(legend_stops)});
  }}
  .legend-tick {{
    position: absolute;
    top: 41px;
    transform: translateX(-50%);
    font-size: 12px;
  }}
  .legend-label {{
    width: 100%;
    text-align: right;
    margin-top: 24px;
    font-size: 12px;
  }}
</style>
</head>
<body>
  <div class="pr-toolbar" id="prZoomBar"></div>
  <div class="plot-scroll" aria-label="{html_module.escape(title)} plot body">
    <div class="chart-grid plot-grid">
      <svg viewBox="0 0 {label_width} {svg_height}" preserveAspectRatio="xMaxYMin meet" aria-hidden="true">
        {"".join(y_labels)}
      </svg>
      <svg id="prPlot" viewBox="0 0 {plot_width} {svg_height}" preserveAspectRatio="none" role="img" aria-label="{html_module.escape(title)}">
        <g id="prGrid">{"".join(grid_lines)}</g>
        <line x1="0" y1="0" x2="0" y2="{svg_height}" stroke="#888" stroke-width="1" />
        {"".join(points)}
      </svg>
      <div></div>
    </div>
  </div>
  <div class="chart-grid axis-row">
    <div></div>
    <svg viewBox="0 0 {plot_width} {axis_height}" preserveAspectRatio="none" aria-hidden="true">
      <g id="prTicks">{"".join(x_tick_labels)}</g>
      <text class="x-label" x="{plot_width}" y="42" text-anchor="end">Pressure Increase (PSI)</text>
    </svg>
    <div></div>
  </div>
  <div class="legend-wrap">
    <div class="legend-bar"></div>
    {"".join(legend_ticks)}
    <div class="legend-label">Order within Realization</div>
  </div>
<script type="application/json" id="pr-zoom-config">{_json_for_script(zoom_config)}</script>
<script>
{_ZOOM_JS}
</script>
<script>
{_PRESSURE_RANGES_ZOOM_SCRIPT}
</script>
</body>
</html>
"""

    artifact_path = _graphs_dir(helper) / f"{artifact_key}.html"
    artifact_path.write_text(html, encoding="utf-8")

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption="Pressure Ranges plot generated by GIST. Drag across the plot to zoom the pressure axis; double-click to reset.",
        renderer="html",
        path=str(artifact_path),
        contentType="text/html",
        displayOrder=display_order,
        preferredHeight=650,
    )


def save_time_series_quantiles_graph_artifact(
    helper,
    quantiles_df,
    artifact_key,
    display_order,
    title=_TIME_SERIES_QUANTILES_TITLE,
):
    """Write total pressure quantiles as an interactive (zoomable) HTML graph artifact."""
    _save_total_time_series_graph_artifact(
        helper,
        quantiles_df,
        artifact_key,
        display_order,
        title,
        group_column="Percentile",
        mode="quantiles",
        color_column=None,
        color_label="Percentile",
        max_groups=None,
        max_points_per_group=_TOTAL_QUANTILES_MAX_POINTS_PER_GROUP,
        caption="Total pressure quantiles generated by GIST. Drag to zoom, Shift+drag to pan, double-click to reset.",
    )


def save_time_series_spaghetti_graph_artifact(
    helper,
    spaghetti_df,
    artifact_key,
    display_order,
    title=_TIME_SERIES_SPAGHETTI_TITLE,
):
    """Write total pressure spaghetti realizations as an interactive (zoomable) HTML graph artifact."""
    _save_total_time_series_graph_artifact(
        helper,
        spaghetti_df,
        artifact_key,
        display_order,
        title,
        group_column="Realization",
        mode="spaghetti",
        color_column="Diffusivity",
        color_label="Diffusivity",
        max_groups=_TOTAL_SPAGHETTI_MAX_REALIZATIONS,
        max_points_per_group=_TOTAL_SPAGHETTI_MAX_POINTS_PER_GROUP,
        caption="Total pressure spaghetti plot generated by GIST. Drag to zoom, Shift+drag to pan, double-click to reset.",
    )


def _save_total_time_series_graph_artifact(
    helper,
    df,
    artifact_key,
    display_order,
    title,
    group_column,
    mode,
    color_column,
    color_label,
    max_groups,
    max_points_per_group,
    caption,
):
    """Render an all-well time series with the shared time series template (no selector, no disposal)."""
    series = _build_time_series_payload(
        df,
        group_column=group_column,
        color_column=color_column,
        max_groups=max_groups,
        max_points_per_group=max_points_per_group,
    )
    if not series:
        return

    color_values = [s["colorValue"] for s in series if s["colorValue"] is not None]
    default_max = 100.0 if mode == "quantiles" else 1.0
    color_norm = _build_time_series_norm(color_values, default_min=0.0, default_max=default_max)

    if mode == "quantiles":
        # The middle percentile is not always exactly 50 (e.g. 49.0 with 11 quantiles), so
        # emphasize the one closest to the median rather than requiring an exact match.
        ranked = [s for s in series if s["colorValue"] is not None]
        if ranked:
            min(ranked, key=lambda s: abs(s["colorValue"] - 50.0))["emphasis"] = True
        line_width, line_opacity = 1.0, 0.95
    else:
        line_width, line_opacity = 0.7, 0.38

    payload = {
        "mode": mode,
        "colorDomain": [float(color_norm.vmin), float(color_norm.vmax)],
        "colorLabel": color_label,
        "showSelector": False,
        "showDisposal": False,
        "yStartAtZero": True,
        "lineWidth": line_width,
        "lineOpacity": line_opacity,
        "wells": [{"key": "total", "label": title, "series": series, "disposal": []}],
    }
    _write_time_series_html_artifact(helper, payload, artifact_key, display_order, title, caption)


def _write_time_series_html_artifact(helper, payload, artifact_key, display_order, title, caption):
    artifact_html = (
        _with_zoom(_TIME_SERIES_HTML_TEMPLATE)
        .replace("___HTML_TITLE___", html_module.escape(title))
        .replace("___PAYLOAD___", _json_for_script(payload))
    )

    artifact_path = _graphs_dir(helper) / f"{artifact_key}.html"
    artifact_path.write_text(artifact_html, encoding="utf-8")

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption=caption,
        renderer="html",
        path=str(artifact_path),
        contentType="text/html",
        displayOrder=display_order,
        preferredHeight=560,
    )


_TIME_SERIES_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>___HTML_TITLE___</title>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #fff; overflow: hidden; }
  #wrap { box-sizing: border-box; height: 100%; padding: 10px 12px 12px; display: flex; flex-direction: column; }
  #toolbar { flex: 0 0 auto; display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 8px 14px; margin-bottom: 8px; font-size: 13px; }
  #colorbar { display: none; align-items: center; gap: 6px; font-size: 12px; margin-right: auto; }
  #colorbar .cb-bar { width: 160px; height: 12px; border: 1px solid #555; background: linear-gradient(to right, #5BB8DC, #000000, #E52327); }
  #colorbar .cb-label { font-weight: 600; }
  #wellPicker { display: inline-flex; align-items: center; gap: 8px; }
  #wellSelect { max-width: min(520px, 78vw); padding: 4px 8px; font: inherit; }
  #svgHost { flex: 1 1 auto; min-height: 240px; position: relative; }
  svg { display: block; width: 100%; height: 100%; user-select: none; }
  .axis-label { font-size: 13px; font-weight: 600; fill: #111; }
  .tick { fill: #333; font-size: 11px; }
  #empty { display: none; padding: 30px; text-align: center; color: #555; }
___ZOOM_CSS___
</style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <div id="colorbar"><span class="cb-label"></span><span class="cb-min"></span><span class="cb-bar"></span><span class="cb-max"></span></div>
    <div id="wellPicker">
      <label for="wellSelect">Well</label>
      <select id="wellSelect"></select>
    </div>
    <div id="zoomBar"></div>
  </div>
  <div id="svgHost">
    <svg id="chart" viewBox="0 0 1000 520" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg"></svg>
    <div id="empty">No data available for this well.</div>
  </div>
</div>
<script type="application/json" id="payload">___PAYLOAD___</script>
<script>
___ZOOM_JS___
</script>
<script>
(function () {
  const payload = JSON.parse(document.getElementById('payload').textContent);
  const select = document.getElementById('wellSelect');
  const svg = document.getElementById('chart');
  const empty = document.getElementById('empty');
  const NS = 'http://www.w3.org/2000/svg';
  const showSelector = payload.showSelector !== false;
  const showDisposal = payload.showDisposal !== false;
  const W = 1000, H = 520;
  const ml = 78, mr = showDisposal ? 78 : 24, mt = 18, mb = 58;
  const plotW = W - ml - mr, plotH = H - mt - mb;
  const x0 = ml, x1 = ml + plotW, y0 = mt, y1 = mt + plotH;
  const DAY_MS = 86400000;

  function clearSvg() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }
  function makeEl(name, attrs) {
    const el = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }
  function fmtDate(ms, spanMs) {
    /* 6 tick intervals: once they are shorter than a month, month labels would repeat, so show days. */
    const opts = spanMs / 6 < 32 * DAY_MS ?{ year: 'numeric', month: 'short', day: 'numeric' } : { year: 'numeric', month: 'short' };
    return new Date(ms).toLocaleDateString(undefined, opts);
  }
  function fmtNum(v) {
    if (!isFinite(v)) return '';
    if (Math.abs(v) >= 100) return Math.round(v).toString();
    if (Math.abs(v) >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
  function fmtLegendValue(v) {
    if (!isFinite(v)) return '';
    return String(Number(v.toPrecision(3)));
  }
  function hexToRgb(hex) {
    const normalized = String(hex || '').replace('#', '');
    if (normalized.length !== 6) return { r: 91, g: 184, b: 220 };
    return {
      r: parseInt(normalized.slice(0, 2), 16),
      g: parseInt(normalized.slice(2, 4), 16),
      b: parseInt(normalized.slice(4, 6), 16)
    };
  }
  function rgbToHex(rgb) {
    const toHex = function (value) {
      return Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0');
    };
    return '#' + toHex(rgb.r) + toHex(rgb.g) + toHex(rgb.b);
  }
  function lerpColor(fromHex, toHex, t) {
    const from = hexToRgb(fromHex);
    const to = hexToRgb(toHex);
    return rgbToHex({
      r: from.r + (to.r - from.r) * t,
      g: from.g + (to.g - from.g) * t,
      b: from.b + (to.b - from.b) * t
    });
  }
  function colorForValue(value, index, count) {
    const domain = payload.colorDomain || [0, 1];
    if (value === null || value === undefined || !isFinite(value)) {
      const fallbackT = count <= 1 ? 0 : index / (count - 1);
      return colorForValue(domain[0] + fallbackT * (domain[1] - domain[0]), index, count);
    }
    let t = domain[1] === domain[0] ? 0 : (value - domain[0]) / (domain[1] - domain[0]);
    t = Math.max(0, Math.min(1, t));
    if (t <= 0.5) {
      return lerpColor('#5BB8DC', '#000000', t / 0.5);
    }
    return lerpColor('#000000', '#E52327', (t - 0.5) / 0.5);
  }
  function roundedBpdAxisMax(maxValue) {
    const increment = maxValue <= 50000 ? 5000 : 10000;
    if (!isFinite(maxValue) || maxValue <= 0) return increment;
    return Math.max(increment, Math.ceil(maxValue / increment) * increment);
  }

  /* Full (unzoomed) extent of a well's pressure series and disposal dates. */
  function computeFull(well) {
    let xmin = Infinity, xmax = -Infinity, ymin = 0, ymax = -Infinity, bpdMax = 0;
    well.series.forEach(function (s) {
      (s.points || []).forEach(function (p) {
        if (p[0] < xmin) xmin = p[0];
        if (p[0] > xmax) xmax = p[0];
        if (p[1] < ymin) ymin = p[1];
        if (p[1] > ymax) ymax = p[1];
      });
    });
    (well.disposal || []).forEach(function (p) {
      if (p[0] < xmin) xmin = p[0];
      if (p[0] > xmax) xmax = p[0];
      if (p[1] > bpdMax) bpdMax = p[1];
    });
    if (!isFinite(xmin) || !isFinite(xmax) || xmin === xmax) { xmin = Date.now() - DAY_MS; xmax = Date.now(); }
    if (!isFinite(ymax) || ymin === ymax) { ymin = 0; ymax = 1; }
    if (!isFinite(bpdMax) || bpdMax <= 0) { bpdMax = 0; }
    const ypad = (ymax - ymin) * 0.06 || 1;
    ymin = payload.yStartAtZero ? Math.min(0, ymin) : Math.min(0, ymin - ypad);
    ymax = ymax + ypad;
    return { xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax, bpdMax: bpdMax };
  }

  let current = null;

  function draw(v) {
    if (!current) return;
    const well = current.well;
    clearSvg();

    const bpdAxisMax = roundedBpdAxisMax(current.full.bpdMax);
    const bpdIncrement = bpdAxisMax <= 50000 ? 5000 : 10000;
    const bpdTickCount = Math.max(1, Math.round(bpdAxisMax / bpdIncrement));

    function sx(t) { return x0 + (t - v.xmin) / (v.xmax - v.xmin) * plotW; }
    function sy(p) { return y1 - (p - v.ymin) / (v.ymax - v.ymin) * plotH; }
    function syBpd(b) { return y1 - (b / bpdAxisMax) * plotH; }

    const defs = makeEl('defs', {});
    const clip = makeEl('clipPath', { id: 'tsPlotClip' });
    clip.appendChild(makeEl('rect', { x: x0, y: y0, width: plotW, height: plotH }));
    defs.appendChild(clip);
    svg.appendChild(defs);
    const dataG = makeEl('g', { 'clip-path': 'url(#tsPlotClip)' });

    const disposal = showDisposal ? (well.disposal || []) : [];
    if (disposal.length) {
      const sortedDisposal = disposal.slice().sort(function (a, b) { return a[0] - b[0]; });
      let barW = plotW / Math.max(80, sortedDisposal.length);
      if (sortedDisposal.length > 1) {
        let minStep = Infinity;
        for (let i = 1; i < sortedDisposal.length; i++) {
          minStep = Math.min(minStep, sx(sortedDisposal[i][0]) - sx(sortedDisposal[i - 1][0]));
        }
        if (isFinite(minStep) && minStep > 0) barW = Math.max(1, Math.min(8, minStep * 0.82));
      }
      sortedDisposal.forEach(function (p) {
        const x = sx(p[0]) - barW / 2;
        if (x + barW < x0 || x > x1) return;
        const y = syBpd(p[1]);
        dataG.appendChild(makeEl('rect', {
          x: x.toFixed(2),
          y: y.toFixed(2),
          width: barW.toFixed(2),
          height: Math.max(0, y1 - y).toFixed(2),
          fill: '#17479E',
          opacity: 0.5
        }));
      });
    }

    const spanMs = v.xmax - v.xmin;
    for (let i = 0; i <= 6; i++) {
      const x = x0 + i / 6 * plotW;
      svg.appendChild(makeEl('line', { x1: x, x2: x, y1: y0, y2: y1, stroke: '#e7e7e7', 'stroke-width': 1 }));
      const t = makeEl('text', { x: x, y: y1 + 20, 'text-anchor': 'middle', class: 'tick' });
      t.textContent = fmtDate(v.xmin + i / 6 * spanMs, spanMs);
      svg.appendChild(t);
    }
    const yTicks = GistZoom.ticks(v.ymin, v.ymax, 6);
    yTicks.values.forEach(function (val) {
      const y = sy(val);
      svg.appendChild(makeEl('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: '#e7e7e7', 'stroke-width': 1 }));
      const t = makeEl('text', { x: x0 - 8, y: y + 4, 'text-anchor': 'end', class: 'tick' });
      t.textContent = GistZoom.fmtTick(val, yTicks.step);
      svg.appendChild(t);
    });
    if (showDisposal) {
      for (let k = 0; k <= bpdTickCount; k++) {
        const val = k * bpdIncrement;
        const y = syBpd(val);
        svg.appendChild(makeEl('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: '#dce8f5', 'stroke-width': 1 }));
        const t = makeEl('text', { x: x1 + 8, y: y + 4, 'text-anchor': 'start', class: 'tick' });
        t.textContent = fmtNum(val);
        svg.appendChild(t);
      }
    }

    const isSpaghetti = payload.mode === 'spaghetti';
    const baseWidth = typeof payload.lineWidth === 'number' ? payload.lineWidth : (isSpaghetti ? 0.9 : 1.8);
    const baseOpacity = typeof payload.lineOpacity === 'number' ? payload.lineOpacity : (isSpaghetti ? 0.34 : 0.94);
    const seriesCount = well.series.length;
    well.series.forEach(function (s, idx) {
      let d = '';
      (s.points || []).forEach(function (p, pointIdx) {
        d += (pointIdx ? ' L ' : 'M ') + sx(p[0]).toFixed(2) + ' ' + sy(p[1]).toFixed(2);
      });
      dataG.appendChild(makeEl('path', {
        d: d,
        fill: 'none',
        stroke: colorForValue(s.colorValue, idx, seriesCount),
        'stroke-width': s.emphasis ? 1.8 : baseWidth,
        opacity: baseOpacity
      }));
    });
    svg.appendChild(dataG);
    svg.appendChild(makeEl('rect', { x: x0, y: y0, width: plotW, height: plotH, fill: 'none', stroke: '#333', 'stroke-width': 1.1 }));

    const xl = makeEl('text', { x: (x0 + x1) / 2, y: H - 14, 'text-anchor': 'middle', class: 'axis-label' });
    xl.textContent = 'Date';
    svg.appendChild(xl);
    const yl = makeEl('text', { x: x0 - 50, y: (y0 + y1) / 2, 'text-anchor': 'middle', class: 'axis-label', transform: 'rotate(-90 ' + (x0 - 50) + ' ' + ((y0 + y1) / 2) + ')' });
    yl.textContent = 'Delta Pressure (PSI)';
    svg.appendChild(yl);
    if (showDisposal) {
      const yr = makeEl('text', { x: x1 + 58, y: (y0 + y1) / 2, 'text-anchor': 'middle', class: 'axis-label', transform: 'rotate(90 ' + (x1 + 58) + ' ' + ((y0 + y1) / 2) + ')' });
      yr.textContent = 'BBL/day';
      svg.appendChild(yr);
    }

    if (payload.mode === 'quantiles' && !payload.colorLabel) {
      const legend = makeEl('g', {});
      const maxItems = Math.min(6, well.series.length);
      for (let i = 0; i < maxItems; i++) {
        const s = well.series[i];
        const lx = x1 - 130 + (i % 2) * 68;
        const ly = y0 + 16 + Math.floor(i / 2) * 18;
        legend.appendChild(makeEl('line', { x1: lx, x2: lx + 18, y1: ly, y2: ly, stroke: colorForValue(s.colorValue, i, well.series.length), 'stroke-width': 2 }));
        const text = makeEl('text', { x: lx + 23, y: ly + 4, class: 'tick' });
        text.textContent = s.label;
        legend.appendChild(text);
      }
      svg.appendChild(legend);
    }
  }

  const zoom = GistZoom.attach({
    svg: svg,
    plot: { x0: x0, y0: y0, x1: x1, y1: y1 },
    full: { xmin: 0, xmax: 1, ymin: 0, ymax: 1 },
    axes: 'xy',
    onChange: draw
  });
  GistZoom.addButtons(document.getElementById('zoomBar'), zoom);

  function render(wellKey) {
    const well = (payload.wells || []).find(function (w) { return w.key === wellKey; });
    if (!well || !well.series || !well.series.length) {
      current = null;
      clearSvg();
      empty.style.display = 'block';
      svg.style.display = 'none';
      return;
    }
    empty.style.display = 'none';
    svg.style.display = 'block';
    const full = computeFull(well);
    current = { well: well, full: full };
    zoom.setFull(full);
  }

  if (payload.colorLabel) {
    const cb = document.getElementById('colorbar');
    const domain = payload.colorDomain || [0, 1];
    cb.querySelector('.cb-label').textContent = payload.colorLabel;
    cb.querySelector('.cb-min').textContent = fmtLegendValue(domain[0]);
    cb.querySelector('.cb-max').textContent = fmtLegendValue(domain[1]);
    cb.style.display = 'inline-flex';
  }
  if (!showSelector) document.getElementById('wellPicker').style.display = 'none';

  (payload.wells || []).forEach(function (well) {
    const opt = document.createElement('option');
    opt.value = well.key;
    opt.textContent = well.label;
    select.appendChild(opt);
  });
  select.addEventListener('change', function () { render(select.value); });
  if (select.options.length) render(select.value);
})();
</script>
</body>
</html>
"""


def _subsample_spaghetti_plot_df(plot_df, group_column):
    """Reduce rows before HTML payload build; min-max downsampling keeps curve shape."""
    if group_column not in plot_df.columns:
        return plot_df

    unique_groups = plot_df[group_column].dropna().unique()
    if len(unique_groups) <= _PER_WELL_SPAGHETTI_MAX_REALIZATIONS:
        return plot_df

    keep_groups = unique_groups[
        np.linspace(0, len(unique_groups) - 1, _PER_WELL_SPAGHETTI_MAX_REALIZATIONS, dtype=int)
    ]
    return plot_df[plot_df[group_column].isin(keep_groups)]


def _save_per_well_time_series_graph_artifact(
    helper,
    df,
    disposal_df,
    artifact_key,
    display_order,
    title,
    group_column,
    mode,
    color_column=None,
):
    plot_df = _coerce_time_series_df(df, ["Date", "DeltaPressure", "subgraph", group_column])
    if plot_df.empty:
        return
    if mode == "spaghetti":
        plot_df = _subsample_spaghetti_plot_df(plot_df, group_column)

    disposal_by_well = _build_disposal_payload(disposal_df)
    wells = []
    for well_name, well_df in plot_df.groupby("subgraph", sort=True):
        if mode == "spaghetti":
            max_groups = _PER_WELL_SPAGHETTI_MAX_GROUPS
            max_points = _PER_WELL_SPAGHETTI_MAX_POINTS_PER_GROUP
        else:
            max_groups = None
            max_points = _PER_WELL_QUANTILES_MAX_POINTS_PER_GROUP
        series = _build_time_series_payload(
            well_df,
            group_column=group_column,
            color_column=color_column,
            max_groups=max_groups,
            max_points_per_group=max_points,
        )
        if not series:
            continue
        wells.append(
            {
                "key": str(well_name),
                "label": str(well_name),
                "series": series,
                "disposal": disposal_by_well.get(str(well_name), []),
            }
        )

    if not wells:
        return

    color_values = []
    for well in wells:
        for series in well["series"]:
            if series["colorValue"] is not None:
                color_values.append(series["colorValue"])
    if mode == "quantiles" and not color_values:
        color_values = [0.0, 100.0]
    color_norm = _build_time_series_norm(color_values, default_min=0.0, default_max=100.0 if mode == "quantiles" else 1.0)

    payload = {
        "mode": mode,
        "colorDomain": [float(color_norm.vmin), float(color_norm.vmax)],
        "wells": wells,
    }
    _write_time_series_html_artifact(
        helper,
        payload,
        artifact_key,
        display_order,
        title,
        caption=(
            "Interactive per-well time series generated by GIST. "
            "Drag to zoom, Shift+drag to pan, double-click to reset."
        ),
    )


def save_time_series_quantiles_per_well_graph_artifact(
    helper,
    quantiles_df,
    disposal_df,
    artifact_key,
    display_order,
    title=_TIME_SERIES_QUANTILES_PER_WELL_TITLE,
):
    """Write per-well pressure quantiles as a standalone HTML graph artifact with a well selector."""
    _save_per_well_time_series_graph_artifact(
        helper,
        quantiles_df,
        disposal_df,
        artifact_key,
        display_order,
        title,
        group_column="Percentile",
        mode="quantiles",
    )


def save_time_series_spaghetti_per_well_graph_artifact(
    helper,
    spaghetti_df,
    disposal_df,
    artifact_key,
    display_order,
    title=_TIME_SERIES_SPAGHETTI_PER_WELL_TITLE,
):
    """Write per-well pressure spaghetti as a standalone HTML graph artifact with a well selector."""
    _save_per_well_time_series_graph_artifact(
        helper,
        spaghetti_df,
        disposal_df,
        artifact_key,
        display_order,
        title,
        group_column="Realization",
        mode="spaghetti",
        color_column="Diffusivity",
    )
