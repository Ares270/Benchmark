"""Shared visual language for self-contained benchmark HTML reports.

Only presentation lives here. No scientific values, thresholds, ordering, or
statistics are computed or transformed by this module.
"""

from __future__ import annotations

import html
from typing import Any


PALETTE = ("#0f766e", "#2563eb", "#7c3aed", "#d97706", "#db2777", "#0891b2")


def report_css() -> str:
    """Return the offline, responsive, print-friendly benchmark stylesheet."""

    return r"""
:root {
  --ink: #172033;
  --muted: #657186;
  --soft: #8a95a8;
  --line: #dfe5ec;
  --line-strong: #cbd5e1;
  --paper: #ffffff;
  --canvas: #f4f7fa;
  --teal: #0f766e;
  --teal-dark: #115e59;
  --teal-soft: #e6f4f1;
  --blue: #2563eb;
  --blue-soft: #eaf1ff;
  --amber: #b45309;
  --amber-soft: #fff7e8;
  --red: #b42318;
  --red-soft: #fff0ee;
  --shadow-sm: 0 1px 2px rgba(16, 24, 40, .05);
  --shadow-md: 0 14px 34px rgba(23, 32, 51, .08);
  --radius: 16px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; background: var(--canvas); }
body {
  color: var(--ink);
  background:
    radial-gradient(circle at 8% 0%, rgba(15,118,110,.10), transparent 26rem),
    radial-gradient(circle at 96% 3%, rgba(37,99,235,.08), transparent 28rem),
    var(--canvas);
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.62;
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 24px 72px;
  text-rendering: optimizeLegibility;
}
body > h1, .report > h1 {
  position: relative;
  margin: 22px 0 10px;
  padding: 30px 32px 28px;
  overflow: hidden;
  color: #fff;
  background: linear-gradient(125deg, #102a3a 0%, #124f59 52%, #0f766e 100%);
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 22px;
  box-shadow: var(--shadow-md);
  font-size: clamp(1.75rem, 4vw, 2.7rem);
  line-height: 1.12;
  letter-spacing: -.035em;
}
body > h1::before, .report > h1::before {
  content: "DYRK1A  /  GENERATIVE BENCHMARK";
  display: block;
  margin-bottom: 13px;
  color: #9de4d9;
  font-size: .72rem;
  font-weight: 750;
  letter-spacing: .16em;
}
body > h1::after, .report > h1::after {
  content: "";
  position: absolute;
  width: 220px;
  height: 220px;
  right: -70px;
  top: -100px;
  border: 36px solid rgba(255,255,255,.08);
  border-radius: 50%;
}
h2 {
  margin: 42px 0 14px;
  padding-bottom: 10px;
  color: #173b48;
  border-bottom: 1px solid var(--line-strong);
  font-size: 1.28rem;
  line-height: 1.3;
  letter-spacing: -.012em;
}
h3 { margin: 28px 0 10px; color: #244654; font-size: 1.02rem; }
p { max-width: 92ch; }
a { color: var(--teal-dark); text-decoration-thickness: .08em; text-underline-offset: .15em; }
a:hover { color: var(--blue); }
code, .mono {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .86em;
  overflow-wrap: anywhere;
}
code { padding: .16rem .38rem; color: #22414d; background: #eaf0f4; border-radius: 6px; }
.report-toolbar {
  position: sticky;
  z-index: 20;
  top: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: min(100%, 1180px);
  margin: 0 auto 12px;
  padding: 9px 10px 9px 15px;
  color: #425466;
  background: rgba(255,255,255,.88);
  border: 1px solid rgba(203,213,225,.88);
  border-radius: 13px;
  box-shadow: 0 10px 28px rgba(23,32,51,.08);
  backdrop-filter: blur(12px);
  font-size: .82rem;
  font-weight: 650;
}
.report-toolbar .brand { display: flex; align-items: center; gap: 9px; }
.report-toolbar .brand::before {
  content: "";
  width: 9px;
  height: 9px;
  background: var(--teal);
  border-radius: 50%;
  box-shadow: 0 0 0 4px var(--teal-soft);
}
.report-actions { display: flex; gap: 7px; }
.report-actions button {
  padding: 7px 11px;
  color: #fff;
  background: var(--teal);
  border: 0;
  border-radius: 9px;
  font: inherit;
  cursor: pointer;
}
.report-actions button:hover { background: var(--teal-dark); }
.sub, .note { color: var(--muted); font-size: .9rem; }
.card, .plotcard {
  position: relative;
  overflow-x: auto;
  margin: 17px 0;
  padding: 18px 20px;
  background: rgba(255,255,255,.96);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}
.card:hover, .plotcard:hover { border-color: #c5d3dc; box-shadow: 0 7px 20px rgba(23,32,51,.06); }
.metrics, .tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
  gap: 13px;
  margin: 21px 0;
}
.metric, .tile {
  min-height: 112px;
  padding: 16px 17px;
  background: linear-gradient(155deg, #fff 20%, #f5faf9 100%);
  border: 1px solid #d8e6e3;
  border-top: 4px solid var(--teal);
  border-radius: 14px;
  box-shadow: var(--shadow-sm);
  color: var(--muted);
  font-size: .77rem;
  font-weight: 750;
  letter-spacing: .045em;
  text-transform: uppercase;
}
.metric:nth-child(3n+2), .tile:nth-child(3n+2) { border-top-color: var(--blue); background: linear-gradient(155deg,#fff 20%,#f4f7ff 100%); }
.metric:nth-child(3n), .tile:nth-child(3n) { border-top-color: #7c3aed; background: linear-gradient(155deg,#fff 20%,#f8f5ff 100%); }
.metric strong, .tile .value {
  display: block;
  margin-top: 8px;
  color: var(--ink);
  font-size: 1.48rem;
  font-weight: 760;
  line-height: 1.1;
  letter-spacing: -.025em;
  text-transform: none;
  font-variant-numeric: tabular-nums;
}
.tile .label { color: var(--muted); font-size: .72rem; }
.tile .ci { margin-top: 7px; color: var(--muted); font-size: .72rem; font-weight: 500; letter-spacing: 0; text-transform: none; }
.warning {
  margin: 18px 0;
  padding: 14px 16px;
  color: #73400b;
  background: var(--amber-soft);
  border: 1px solid #f3d29b;
  border-left: 5px solid #d97706;
  border-radius: 12px;
}
table.data, table.kv, table.comparison {
  width: 100%;
  border-spacing: 0;
  border-collapse: separate;
  overflow: hidden;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  font-size: .88rem;
  font-variant-numeric: tabular-nums;
}
table.data th, table.data td,
table.kv th, table.kv td,
table.comparison th, table.comparison td {
  padding: 10px 12px;
  border: 0;
  border-bottom: 1px solid #edf1f5;
  text-align: right;
  white-space: nowrap;
}
table.data th, table.kv th, table.comparison th {
  color: #455366;
  background: #eef3f6;
  font-size: .74rem;
  font-weight: 760;
  letter-spacing: .045em;
  text-transform: uppercase;
}
table.data tr:nth-child(even) td,
table.kv tr:nth-child(even) td,
table.comparison tr:nth-child(even) td { background: #fafcfd; }
table.data tr:last-child td, table.kv tr:last-child td, table.comparison tr:last-child td { border-bottom: 0; }
table.data th:first-child, table.data td:first-child,
table.kv th:first-child, table.kv td:first-child,
table.comparison th:first-child, table.comparison td:first-child { text-align: left; }
.cap { margin: 2px 5px 9px; color: var(--muted); font-size: .84rem; }
.foot {
  margin-top: 46px;
  padding: 18px 20px;
  color: var(--muted);
  background: #eaf0f4;
  border-radius: var(--radius);
  font-size: .82rem;
}
.section-heading-row {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 22px;
  margin-top: 42px;
  border-bottom: 1px solid var(--line-strong);
}
.section-heading-row h2 { margin: 0; padding: 0 0 10px; border: 0; }
.section-heading-row p { margin: 0 0 11px; color: var(--muted); font-size: .88rem; }
.gallery-tabs {
  display: inline-flex;
  flex: 0 0 auto;
  gap: 4px;
  margin-bottom: 10px;
  padding: 4px;
  background: #e7edf1;
  border-radius: 11px;
}
.gallery-tabs button {
  padding: 7px 13px;
  color: #526174;
  background: transparent;
  border: 0;
  border-radius: 8px;
  font: inherit;
  font-size: .8rem;
  font-weight: 720;
  cursor: pointer;
}
.gallery-tabs button.active {
  color: #fff;
  background: var(--teal);
  box-shadow: var(--shadow-sm);
}
.gallery-panel { display: none; }
.gallery-panel.active { display: block; }
.molecule-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(225px, 1fr));
  gap: 14px;
  margin: 18px 0;
}
.molecule-card {
  min-width: 0;
  overflow: hidden;
  background: rgba(255,255,255,.97);
  border: 1px solid var(--line);
  border-radius: 15px;
  box-shadow: var(--shadow-sm);
  transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}
.molecule-card:hover {
  transform: translateY(-2px);
  border-color: #b8cbd2;
  box-shadow: 0 12px 28px rgba(23,32,51,.10);
}
.molecule-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px 0;
}
.molecule-edge {
  padding: 4px 7px;
  border-radius: 99px;
  font-size: .66rem;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.molecule-edge.best { color: #0b625b; background: var(--teal-soft); }
.molecule-edge.worst { color: #8c3f07; background: var(--amber-soft); }
.molecule-rank { color: var(--muted); font-size: .7rem; font-variant-numeric: tabular-nums; }
.molecule-drawing {
  display: grid;
  place-items: center;
  min-height: 175px;
  margin: 5px 8px 0;
  overflow: hidden;
  background: #fff;
  border-radius: 10px;
}
.molecule-drawing svg { display: block; width: 100%; height: auto; max-height: 180px; }
.molecule-missing { color: var(--muted); font-size: .8rem; }
.molecule-id {
  padding: 7px 13px 0;
  overflow: hidden;
  color: #26394a;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: .73rem;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.molecule-score {
  padding: 4px 13px 9px;
  color: var(--ink);
  font-size: 1.35rem;
  font-weight: 770;
  font-variant-numeric: tabular-nums;
}
.molecule-score small { margin-right: 8px; color: var(--muted); font-size: .63rem; letter-spacing: .06em; text-transform: uppercase; }
.molecule-score em { color: var(--muted); font-size: .62rem; font-style: normal; font-weight: 600; }
.molecule-stats { display: grid; grid-template-columns: repeat(4,1fr); border-top: 1px solid #edf1f5; }
.molecule-stat { padding: 8px 5px; text-align: center; font-size: .75rem; font-weight: 720; font-variant-numeric: tabular-nums; }
.molecule-stat small { display: block; color: var(--muted); font-size: .58rem; font-weight: 650; text-transform: uppercase; }
.smiles-details { padding: 8px 12px 11px; color: var(--muted); border-top: 1px solid #edf1f5; font-size: .7rem; }
.smiles-details summary { cursor: pointer; font-weight: 680; }
.smiles-details code { display: block; margin-top: 7px; white-space: normal; }
.js-plotly-plot, .plot-container { width: 100% !important; }
@media (max-width: 720px) {
  body { padding: 16px 12px 52px; font-size: 14px; }
  body > h1, .report > h1 { padding: 25px 22px; border-radius: 17px; }
  .report-toolbar { top: 6px; }
  .report-toolbar .brand span { display: none; }
  .metrics, .tiles { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .metric, .tile { min-height: 96px; padding: 13px; }
  table.data th, table.data td, table.kv th, table.kv td { padding: 8px 9px; }
  .section-heading-row { align-items: stretch; flex-direction: column; gap: 4px; }
  .gallery-tabs { align-self: flex-start; }
}
@media (max-width: 430px) { .metrics, .tiles { grid-template-columns: 1fr; } }
@media print {
  html, body { background: #fff !important; }
  body { max-width: none; padding: 0; color: #111; }
  .report-toolbar { display: none; }
  body > h1, .report > h1 { color: #111; background: #fff; border: 2px solid #333; box-shadow: none; }
  body > h1::before, .report > h1::before { color: #333; }
  .card, .plotcard, .metric, .tile { break-inside: avoid; box-shadow: none; }
  h2 { break-after: avoid; }
}
"""


def report_toolbar(label: str = "Benchmark report") -> str:
    """Small offline toolbar with print/PDF and top actions."""

    safe = html.escape(label)
    return (
        '<div class="report-toolbar"><div class="brand"><span>'
        f"{safe}</span></div><div class=\"report-actions\">"
        '<button type="button" onclick="window.scrollTo({top:0,behavior:\'smooth\'})">Top</button>'
        '<button type="button" onclick="window.print()">Print / PDF</button>'
        "</div></div>"
    )


def plotly_config() -> dict[str, Any]:
    """Consistent responsive controls for offline Plotly figures."""

    return {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": False,
        "modeBarButtonsToRemove": ("lasso2d", "select2d"),
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }


def polish_plotly_figure(figure: Any, *, height: int | None = None) -> Any:
    """Apply typography/layout styling without changing traces or values."""

    layout: dict[str, Any] = {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#ffffff",
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            "size": 12,
            "color": "#344054",
        },
        "colorway": list(PALETTE),
        "hoverlabel": {"bgcolor": "#172033", "font_color": "#ffffff"},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255,255,255,.72)",
        },
        "margin": {"l": 60, "r": 28, "t": 72, "b": 58},
    }
    if height is not None:
        layout["height"] = height
    figure.update_layout(**layout)
    figure.update_xaxes(
        gridcolor="#edf1f5",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
        title_standoff=12,
    )
    figure.update_yaxes(
        gridcolor="#edf1f5",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
        title_standoff=12,
    )
    return figure
