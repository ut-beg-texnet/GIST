import base64
import html as html_module
import io
import json
import math
import pathlib
from datetime import datetime
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
_TIME_SERIES_DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "texnet_time_series_ramp",
    ["#5BB8DC", "#000000", "#E52327"],
)
_PRESSURE_RANGES_CMAP = plt.get_cmap("viridis")


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
  #legend { display: flex; flex-wrap: wrap; gap: 8px 14px; justify-content: center; align-items: center; font-size: 11px; margin-bottom: 6px; }
  #legend .item { display: inline-flex; align-items: center; gap: 4px; }
  #legend .sw { width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(0,0,0,.75); flex-shrink: 0; }
  #legend .line-swatch { width: 22px; height: 0; flex-shrink: 0; border-top-width: 2px; border-top-style: solid; display: inline-block; vertical-align: middle; }
  #svgHost { flex: 1; min-height: 200px; position: relative; }
  svg { display: block; width: 100%; height: 100%; }
  #tip {
    position: absolute; display: none; pointer-events: none; z-index: 10;
    background: rgba(30, 30, 35, 0.94); color: #f5f5f5; padding: 8px 10px; border-radius: 6px;
    font-size: 12px; line-height: 1.45; max-width: 280px; box-shadow: 0 2px 10px rgba(0,0,0,.2);
  }
  #tip .k { color: #aaa; }
</style>
</head>
<body>
<div id="wrap">
  <div id="legend"></div>
  <div id="svgHost">
    <svg id="chart" viewBox="0 0 1000 560" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
      <rect id="hitLayer" x="0" y="0" width="1000" height="560" fill="transparent" cursor="crosshair"/>
    </svg>
    <div id="tip"></div>
  </div>
</div>
<script type="application/json" id="rt-payload">___PAYLOAD___</script>
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

  function xScale(x) { return x0 + (x - xmin) / (xmax - xmin) * plotW; }
  function yScale(y) { return y1 - (y - ymin) / (ymax - ymin) * plotH; }

  const svg = document.getElementById('chart');
  const NS = 'http://www.w3.org/2000/svg';
  const gridG = document.createElementNS(NS, 'g');
  gridG.setAttribute('class', 'grid');
  const nGridX = 10, nGridY = 8;
  for (var gi = 0; gi <= nGridX; gi++) {
    var gx = x0 + (gi / nGridX) * plotW;
    var line = document.createElementNS(NS, 'line');
    line.setAttribute('x1', gx); line.setAttribute('x2', gx);
    line.setAttribute('y1', y0); line.setAttribute('y2', y1);
    line.setAttribute('stroke', '#e8e8e8'); line.setAttribute('stroke-width', '1');
    gridG.appendChild(line);
  }
  for (var gj = 0; gj <= nGridY; gj++) {
    var gy = y0 + (gj / nGridY) * plotH;
    var line2 = document.createElementNS(NS, 'line');
    line2.setAttribute('x1', x0); line2.setAttribute('x2', x1);
    line2.setAttribute('y1', gy); line2.setAttribute('y2', gy);
    line2.setAttribute('stroke', '#e8e8e8'); line2.setAttribute('stroke-width', '1');
    gridG.appendChild(line2);
  }
  svg.insertBefore(gridG, document.getElementById('hitLayer'));

  const axisG = document.createElementNS(NS, 'g');
  const border = document.createElementNS(NS, 'rect');
  border.setAttribute('x', x0); border.setAttribute('y', y0);
  border.setAttribute('width', plotW); border.setAttribute('height', plotH);
  border.setAttribute('fill', 'none'); border.setAttribute('stroke', '#333');
  border.setAttribute('stroke-width', '1.2');
  axisG.appendChild(border);

  function fmtAxisTick(val) {
    if (!isFinite(val)) return '';
    if (Math.abs(val - Math.round(val)) < 1e-5) return String(Math.round(val));
    var r = Math.round(val * 10) / 10;
    return String(r);
  }
  function makeTickText(val) {
    var t = document.createElementNS(NS, 'text');
    t.setAttribute('fill', '#333');
    t.setAttribute('font-size', '11');
    t.textContent = fmtAxisTick(val);
    return t;
  }
  var nx = 6;
  for (var ti = 0; ti <= nx; ti++) {
    var xv = xmin + (ti / nx) * (xmax - xmin);
    var tx = makeTickText(xv);
    tx.setAttribute('x', xScale(xv));
    tx.setAttribute('y', y1 + 18);
    tx.setAttribute('text-anchor', 'middle');
    axisG.appendChild(tx);
  }
  var ny = 6;
  for (var tj = 0; tj <= ny; tj++) {
    var yv = ymin + (tj / ny) * (ymax - ymin);
    var ty = makeTickText(yv);
    ty.setAttribute('x', x0 - 8);
    ty.setAttribute('y', yScale(yv) + 4);
    ty.setAttribute('text-anchor', 'end');
    axisG.appendChild(ty);
  }
  var xl = document.createElementNS(NS, 'text');
  xl.setAttribute('x', (x0 + x1) / 2); xl.setAttribute('y', H - 10);
  xl.setAttribute('text-anchor', 'middle'); xl.setAttribute('fill', '#111');
  xl.setAttribute('font-size', '13'); xl.setAttribute('font-weight', '600');
  xl.textContent = xLabel;
  axisG.appendChild(xl);
  var yl = document.createElementNS(NS, 'text');
  yl.setAttribute('transform', 'rotate(-90 ' + (x0 - 46) + ' ' + ((y0 + y1) / 2) + ')');
  yl.setAttribute('x', x0 - 46); yl.setAttribute('y', (y0 + y1) / 2);
  yl.setAttribute('text-anchor', 'middle'); yl.setAttribute('fill', '#111');
  yl.setAttribute('font-size', '13'); yl.setAttribute('font-weight', '600');
  yl.textContent = yLabel;
  axisG.appendChild(yl);
  svg.insertBefore(axisG, document.getElementById('hitLayer'));

  /* Circles first, then diffusivity curves, so lines stay in the foreground. */
  const defaultR = (typeof payload.pointRadius === 'number' && isFinite(payload.pointRadius)) ? payload.pointRadius : 5;
  const plotPts = [];
  pts.forEach(function (p) {
    var cx = xScale(p.x), cy = yScale(p.y);
    var pr = (typeof p.r === 'number' && isFinite(p.r) && p.r > 0) ? p.r : defaultR;
    plotPts.push({ cx: cx, cy: cy, r: pr, p: p });
    var c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', cx); c.setAttribute('cy', cy); c.setAttribute('r', String(pr));
    c.setAttribute('fill', p.color || '#333');
    c.setAttribute('stroke', p.stroke || '#000'); c.setAttribute('stroke-width', '1');
    c.setAttribute('opacity', '0.88');
    c.setAttribute('pointer-events', 'none');
    svg.insertBefore(c, document.getElementById('hitLayer'));
  });

  curves.forEach(function (curve) {
    var d = '';
    (curve.points || []).forEach(function (pt, i) {
      var sx = xScale(pt[0]), sy = yScale(pt[1]);
      d += (i ? ' L ' : 'M ') + sx.toFixed(2) + ' ' + sy.toFixed(2) + ' ';
    });
    var path = document.createElementNS(NS, 'path');
    path.setAttribute('d', d.trim());
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', curve.color || '#222');
    path.setAttribute('stroke-width', '2');
    if (curve.dash === 'dash') path.setAttribute('stroke-dasharray', '7 5');
    svg.insertBefore(path, document.getElementById('hitLayer'));
  });

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
    var xv = (typeof p.x === 'number' && isFinite(p.x)) ? p.x.toFixed(3) : String(p.x);
    var yv = (typeof p.y === 'number' && isFinite(p.y)) ? p.y.toFixed(2) : String(p.y);
    var mmbblLine = '';
    if (typeof p.mmbbl === 'number' && isFinite(p.mmbbl)) {
      mmbblLine = '<div><span class="k">MMBBL</span> ' + p.mmbbl.toFixed(4) + '</div>';
    }
    tip.innerHTML =
      '<div><span class="k">UIC Number</span> ' + uicShow + '</div>' +
      mmbblLine +
      '<div><span class="k">' + escapeHtml(xLabel) + '</span> ' + xv + '</div>' +
      '<div><span class="k">' + escapeHtml(yLabel) + '</span> ' + yv + '</div>';
    tip.style.display = 'block';
    var rect = host.getBoundingClientRect();
    var tx = clientX - rect.left + 12, ty = clientY - rect.top + 12;
    tip.style.left = Math.min(tx, rect.width - 200) + 'px';
    tip.style.top = Math.min(ty, rect.height - 80) + 'px';
  }

  function hideTip() { tip.style.display = 'none'; }

  svg.addEventListener('mousemove', function (ev) {
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

  document.getElementById('hitLayer').setAttribute('pointer-events', 'none');
})();
</script>
</body>
</html>
"""


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
        plot_df["timestamp"] = pd.to_datetime(plot_df["Date"], errors="coerce").view("int64") // 10**6
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    plot_df["DeltaPressure"] = pd.to_numeric(plot_df["DeltaPressure"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Date", "DeltaPressure", "timestamp"])
    if plot_df.empty:
        return pd.DataFrame()
    plot_df["timestamp"] = plot_df["timestamp"].astype("int64")
    return plot_df.sort_values("timestamp").reset_index(drop=True)


def _write_matplotlib_png(fig, artifact_path, tight=True):
    save_kwargs = {"format": "png", "dpi": 145, "facecolor": "white"}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(artifact_path, **save_kwargs)
    plt.close(fig)


def _configure_time_series_datetime_axis(ax, plot_df):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.set_xlabel("Date / Time")
    ax.tick_params(axis="x", labelbottom=True, pad=8, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")


def _configure_time_series_y_axis(ax, plot_df):
    y_max = plot_df["DeltaPressure"].max()
    if pd.isna(y_max):
        return

    top_pad = max(y_max * 0.04, 1.0) if y_max > 0 else 1.0
    ax.set_ylim(bottom=0.0, top=y_max + top_pad)
    ax.margins(x=0.02, y=0.0)


def _create_time_series_figure():
    fig = plt.figure(figsize=(11.4, 5.8))
    grid = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[24, 1.5], wspace=0.12)
    ax = fig.add_subplot(grid[0])
    cax = fig.add_subplot(grid[1])
    return fig, ax, cax


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


def _hex_color(color_value):
    return mcolors.to_hex(color_value, keep_alpha=False)


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


def _time_series_color_for_value(value, color_norm):
    if value is None or color_norm is None:
        return "#5BB8DC"
    return _hex_color(_TIME_SERIES_DIVERGING_CMAP(color_norm(float(value))))


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


def _build_disposal_payload(disposal_df):
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
        plot_df["timestamp"] = pd.to_datetime(plot_df["Date"], errors="coerce").view("int64") // 10**6
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
    parsed = pd.to_datetime(well_df["StartDate"], errors="coerce")
    future_mask = parsed.notna() & (parsed.dt.normalize() > ref_day)

    before = len(well_df)
    filtered = well_df[~future_mask].copy()
    if len(filtered) < before:
        print(
            f"R-t plot excluded {before - len(filtered)}/{before} wells "
            f"with StartDate after {ref_day.strftime('%Y-%m-%d')}.",
            flush=True,
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
                "x": float(row["YearsInjectingToEarthquake"]),
                "y": float(row["Distances"]),
                "mmbbl": mmb_f,
                "r": _rt_svg_radius_from_mmbbl(mmb_f),
                "color": _RT_SELECTION_STYLES.get(sel, fallback_style)["fill"],
                "stroke": _RT_SELECTION_STYLES.get(sel, fallback_style)["stroke"],
            }
        )

    curves = []
    for diffusivity, curve_df in small_pp_df.groupby("Diffusivity", sort=False):
        curve_df = curve_df.sort_values("Years Before Earthquake")
        curve_style = _RT_CURVE_STYLES.get(diffusivity, {"color": "#444444", "dash": "solid"})
        xy_pairs = curve_df[["Years Before Earthquake", "Distance"]].to_numpy(dtype=float).tolist()
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
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html_out = _RT_PLOT_HTML_TEMPLATE.replace("___PAYLOAD___", payload_json)
    artifact_path.write_text(html_out, encoding="utf-8")

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption="Interactive R-t plot: marker size reflects MMBBL; hover for UIC number, MMBBL, and axis values.",
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

    label_width = 240
    plot_width = 900
    right_pad = 32
    row_height = 30
    svg_width = label_width + plot_width + right_pad
    svg_height = max(90, well_count * row_height + 16)

    def x_px(value):
        return label_width + (float(value) / x_max) * plot_width

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
        tx = x_px(tick)
        grid_lines.append(
            f'<line x1="{tx:.1f}" y1="0" x2="{tx:.1f}" y2="{svg_height - 16}" stroke="#e5e5e5" stroke-width="1" />'
        )
        x_tick_labels.append(
            f'<span class="x-tick" style="left:{((tx - label_width) / plot_width) * 100:.4f}%">{html_module.escape(str(tick))}</span>'
        )

    y_labels = []
    for name, y_index in y_lookup.items():
        y = y_index * row_height + row_height / 2
        y_labels.append(
            f'<text x="{label_width - 8}" y="{y + 4:.1f}" text-anchor="end">{html_module.escape(str(name))}</text>'
        )

    points = []
    for _, row in plot_df.iterrows():
        x = x_px(row["Pressures"])
        y = float(row["Y"]) * row_height + row_height / 2
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" fill="{color_for_order(row["Order"])}" opacity="0.82" />'
        )

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
  .plot-scroll {{
    flex: 1 1 auto;
    min-height: 220px;
    overflow-y: auto;
    overflow-x: hidden;
    border-bottom: 1px solid #888;
  }}
  svg {{
    display: block;
    box-sizing: border-box;
    width: 100%;
    min-width: {svg_width}px;
    height: {svg_height}px;
  }}
  text {{
    fill: #1f1f1f;
    font-size: 12px;
  }}
  .axis {{
    flex: 0 0 48px;
    margin-left: {label_width}px;
    margin-right: {right_pad}px;
    position: relative;
    border-top: 1px solid #888;
  }}
  .x-tick {{
    position: absolute;
    top: 5px;
    transform: translateX(-50%);
    font-size: 12px;
  }}
  .x-label {{
    position: absolute;
    right: 0;
    bottom: 4px;
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
  <div class="plot-scroll" aria-label="{html_module.escape(title)} plot body">
    <svg viewBox="0 0 {svg_width} {svg_height}" preserveAspectRatio="none" role="img" aria-label="{html_module.escape(title)}">
      {"".join(grid_lines)}
      <line x1="{label_width}" y1="0" x2="{label_width}" y2="{svg_height - 16}" stroke="#888" stroke-width="1" />
      {"".join(y_labels)}
      {"".join(points)}
    </svg>
  </div>
  <div class="axis">
    {"".join(x_tick_labels)}
    <div class="x-label">Pressure Increase (PSI)</div>
  </div>
  <div class="legend-wrap">
    <div class="legend-bar"></div>
    {"".join(legend_ticks)}
    <div class="legend-label">Order within Realization</div>
  </div>
</body>
</html>
"""

    artifact_path = _graphs_dir(helper) / f"{artifact_key}.html"
    artifact_path.write_text(html, encoding="utf-8")

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption="Static Pressure Ranges plot generated by GIST.",
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
    """Write total pressure quantiles as a static PNG graph artifact."""
    plot_df = _coerce_time_series_df(quantiles_df, ["Date", "DeltaPressure", "Percentile"])
    if plot_df.empty:
        return

    artifact_path = _graphs_dir(helper) / f"{artifact_key}.png"
    fig, ax, cax = _create_time_series_figure()
    percentiles = sorted(plot_df["Percentile"].dropna().unique())
    percentile_norm = _build_time_series_norm(percentiles, default_min=0.0, default_max=100.0)
    for idx, percentile in enumerate(percentiles):
        series_df = plot_df[plot_df["Percentile"] == percentile].sort_values("Date")
        percentile_value = _safe_float(percentile)
        ax.plot(
            series_df["Date"],
            series_df["DeltaPressure"],
            linewidth=1.7 if float(percentile) == 50.0 else 1.0,
            color=_time_series_color_for_value(percentile_value, percentile_norm),
            alpha=0.95,
        )

    _configure_time_series_datetime_axis(ax, plot_df)
    _configure_time_series_y_axis(ax, plot_df)
    ax.set_ylabel("Delta Pressure (PSI)")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    sm = plt.cm.ScalarMappable(cmap=_TIME_SERIES_DIVERGING_CMAP, norm=percentile_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Percentile")
    fig.subplots_adjust(left=0.09, right=0.92, bottom=0.2, top=0.98)
    _write_matplotlib_png(fig, artifact_path, tight=False)

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption="Static total pressure quantiles generated by GIST.",
        renderer="image",
        path=str(artifact_path),
        contentType="image/png",
        displayOrder=display_order,
        preferredHeight=520,
    )


def save_time_series_spaghetti_graph_artifact(
    helper,
    spaghetti_df,
    artifact_key,
    display_order,
    title=_TIME_SERIES_SPAGHETTI_TITLE,
):
    """Write total pressure spaghetti realizations as a static PNG graph artifact."""
    plot_df = _coerce_time_series_df(spaghetti_df, ["Date", "DeltaPressure", "Realization"])
    if plot_df.empty:
        return

    artifact_path = _graphs_dir(helper) / f"{artifact_key}.png"
    fig, ax, cax = _create_time_series_figure()

    if "Diffusivity" in plot_df.columns:
        diffusivity_values = pd.to_numeric(plot_df["Diffusivity"], errors="coerce")
    else:
        diffusivity_values = pd.Series(dtype="float64")
    finite_diffusivity = diffusivity_values.dropna()
    if finite_diffusivity.empty:
        norm = _build_time_series_norm([], default_min=0.0, default_max=1.0)
    else:
        norm = _build_time_series_norm(finite_diffusivity.tolist(), default_min=finite_diffusivity.min(), default_max=finite_diffusivity.max())

    for _, series_df in plot_df.groupby("Realization", sort=True):
        series_df = series_df.sort_values("Date")
        color_value = None
        if "Diffusivity" in series_df.columns:
            diff_value = pd.to_numeric(series_df["Diffusivity"], errors="coerce").dropna()
            if not diff_value.empty:
                color_value = float(diff_value.iloc[0])
        color = _time_series_color_for_value(color_value, norm)
        ax.plot(series_df["Date"], series_df["DeltaPressure"], linewidth=0.7, color=color, alpha=0.38)

    _configure_time_series_datetime_axis(ax, plot_df)
    _configure_time_series_y_axis(ax, plot_df)
    ax.set_ylabel("Delta Pressure (PSI)")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    sm = plt.cm.ScalarMappable(cmap=_TIME_SERIES_DIVERGING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Diffusivity")
    fig.subplots_adjust(left=0.09, right=0.92, bottom=0.2, top=0.98)
    _write_matplotlib_png(fig, artifact_path, tight=False)

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption="Static total pressure spaghetti plot generated by GIST.",
        renderer="image",
        path=str(artifact_path),
        contentType="image/png",
        displayOrder=display_order,
        preferredHeight=520,
    )


_PER_WELL_TIME_SERIES_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>___HTML_TITLE___</title>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #fff; overflow: hidden; }
  #wrap { box-sizing: border-box; height: 100%; padding: 10px 12px 12px; display: flex; flex-direction: column; }
  #toolbar { flex: 0 0 auto; display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 13px; }
  #wellSelect { max-width: min(520px, 78vw); padding: 4px 8px; font: inherit; }
  #svgHost { flex: 1 1 auto; min-height: 240px; position: relative; }
  svg { display: block; width: 100%; height: 100%; }
  .axis-label { font-size: 13px; font-weight: 600; fill: #111; }
  .tick { fill: #333; font-size: 11px; }
  #empty { display: none; padding: 30px; text-align: center; color: #555; }
</style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <label for="wellSelect">Well</label>
    <select id="wellSelect"></select>
  </div>
  <div id="svgHost">
    <svg id="chart" viewBox="0 0 1000 520" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg"></svg>
    <div id="empty">No data available for this well.</div>
  </div>
</div>
<script type="application/json" id="payload">___PAYLOAD___</script>
<script>
(function () {
  const payload = JSON.parse(document.getElementById('payload').textContent);
  const select = document.getElementById('wellSelect');
  const svg = document.getElementById('chart');
  const empty = document.getElementById('empty');
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1000, H = 520;
  const ml = 78, mr = 78, mt = 18, mb = 58;
  const plotW = W - ml - mr, plotH = H - mt - mb;
  const x0 = ml, x1 = ml + plotW, y0 = mt, y1 = mt + plotH;

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function clearSvg() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }
  function makeEl(name, attrs) {
    const el = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }
  function fmtDate(ms) {
    return new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short' });
  }
  function fmtNum(v) {
    if (!isFinite(v)) return '';
    if (Math.abs(v) >= 100) return Math.round(v).toString();
    if (Math.abs(v) >= 10) return v.toFixed(1);
    return v.toFixed(2);
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
  function render(wellKey) {
    const well = (payload.wells || []).find(function (w) { return w.key === wellKey; });
    clearSvg();
    if (!well || !well.series || !well.series.length) {
      empty.style.display = 'block';
      svg.style.display = 'none';
      return;
    }
    empty.style.display = 'none';
    svg.style.display = 'block';

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
    if (!isFinite(xmin) || !isFinite(xmax) || xmin === xmax) { xmin = Date.now() - 86400000; xmax = Date.now(); }
    if (!isFinite(ymax) || ymin === ymax) { ymin = 0; ymax = 1; }
    if (!isFinite(bpdMax) || bpdMax <= 0) { bpdMax = 0; }
    const ypad = (ymax - ymin) * 0.06 || 1;
    ymin = Math.min(0, ymin - ypad);
    ymax = ymax + ypad;

    function roundedBpdAxisMax(maxValue) {
      const increment = maxValue <= 50000 ? 5000 : 10000;
      if (!isFinite(maxValue) || maxValue <= 0) return increment;
      return Math.max(increment, Math.ceil(maxValue / increment) * increment);
    }
    const bpdAxisMax = roundedBpdAxisMax(bpdMax);
    const bpdIncrement = bpdAxisMax <= 50000 ? 5000 : 10000;
    const bpdTickCount = Math.max(1, Math.round(bpdAxisMax / bpdIncrement));

    function sx(v) { return x0 + (v - xmin) / (xmax - xmin) * plotW; }
    function sy(v) { return y1 - (v - ymin) / (ymax - ymin) * plotH; }
    function syBpd(v) { return y1 - (v / bpdAxisMax) * plotH; }

    const disposal = well.disposal || [];
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
        const y = syBpd(p[1]);
        svg.appendChild(makeEl('rect', {
          x: x.toFixed(2),
          y: y.toFixed(2),
          width: barW.toFixed(2),
          height: Math.max(0, y1 - y).toFixed(2),
          fill: '#17479E',
          opacity: 0.5
        }));
      });
    }

    for (let i = 0; i <= 6; i++) {
      const x = x0 + i / 6 * plotW;
      svg.appendChild(makeEl('line', { x1: x, x2: x, y1: y0, y2: y1, stroke: '#e7e7e7', 'stroke-width': 1 }));
      const t = makeEl('text', { x: x, y: y1 + 20, 'text-anchor': 'middle', class: 'tick' });
      t.textContent = fmtDate(xmin + i / 6 * (xmax - xmin));
      svg.appendChild(t);
    }
    for (let j = 0; j <= 5; j++) {
      const y = y0 + j / 5 * plotH;
      svg.appendChild(makeEl('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: '#e7e7e7', 'stroke-width': 1 }));
      const v = ymax - j / 5 * (ymax - ymin);
      const t = makeEl('text', { x: x0 - 8, y: y + 4, 'text-anchor': 'end', class: 'tick' });
      t.textContent = fmtNum(v);
      svg.appendChild(t);
    }
    for (let k = 0; k <= bpdTickCount; k++) {
      const v = k * bpdIncrement;
      const y = syBpd(v);
      svg.appendChild(makeEl('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: '#dce8f5', 'stroke-width': 1 }));
      const t = makeEl('text', { x: x1 + 8, y: y + 4, 'text-anchor': 'start', class: 'tick' });
      t.textContent = fmtNum(v);
      svg.appendChild(t);
    }
    svg.appendChild(makeEl('rect', { x: x0, y: y0, width: plotW, height: plotH, fill: 'none', stroke: '#333', 'stroke-width': 1.1 }));

    const seriesCount = well.series.length;
    well.series.forEach(function (s, idx) {
      let d = '';
      (s.points || []).forEach(function (p, pointIdx) {
        d += (pointIdx ? ' L ' : 'M ') + sx(p[0]).toFixed(2) + ' ' + sy(p[1]).toFixed(2);
      });
      svg.appendChild(makeEl('path', {
        d: d,
        fill: 'none',
        stroke: colorForValue(s.colorValue, idx, seriesCount),
        'stroke-width': payload.mode === 'spaghetti' ? 0.9 : 1.8,
        opacity: payload.mode === 'spaghetti' ? 0.34 : 0.94
      }));
    });

    const xl = makeEl('text', { x: (x0 + x1) / 2, y: H - 14, 'text-anchor': 'middle', class: 'axis-label' });
    xl.textContent = 'Date';
    svg.appendChild(xl);
    const yl = makeEl('text', { x: x0 - 50, y: (y0 + y1) / 2, 'text-anchor': 'middle', class: 'axis-label', transform: 'rotate(-90 ' + (x0 - 50) + ' ' + ((y0 + y1) / 2) + ')' });
    yl.textContent = 'Delta Pressure (PSI)';
    svg.appendChild(yl);
    const yr = makeEl('text', { x: x1 + 58, y: (y0 + y1) / 2, 'text-anchor': 'middle', class: 'axis-label', transform: 'rotate(90 ' + (x1 + 58) + ' ' + ((y0 + y1) / 2) + ')' });
    yr.textContent = 'BBL/day';
    svg.appendChild(yr);

    if (payload.mode === 'quantiles') {
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
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    artifact_html = (
        _PER_WELL_TIME_SERIES_HTML_TEMPLATE
        .replace("___HTML_TITLE___", html_module.escape(title))
        .replace("___PAYLOAD___", payload_json)
    )

    artifact_path = _graphs_dir(helper) / f"{artifact_key}.html"
    artifact_path.write_text(artifact_html, encoding="utf-8")

    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        caption="Interactive per-well time series generated by GIST.",
        renderer="html",
        path=str(artifact_path),
        contentType="text/html",
        displayOrder=display_order,
        preferredHeight=560,
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
