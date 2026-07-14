"""Create a scale-honest comparison report from multiple metrics.json files.

This is the tail end of the pipe

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

from . import plots

COMPARE_METRICS = ("auc", "bedroc", "ef_1pct", "ef_5pct", "ef_10pct")     # THIS is the whitelist, nothing else matters





#############    Take JSON file from run and strip it down to single row dictionary   ###########

def load_run(metrics_json: Path) -> dict:
    path = Path(metrics_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError(f"{path} has no 'metrics' object")                 # Report broken metrics 
    row = {"method": str(data.get("name") or path.parent.name)}             # Label the Run with contents of JSON
    row.update(data["metrics"])                                             # or with name of parent dir
    return row







#######   Build from the previous JSON payload   #######

def build_comparison_table(metrics_jsons: list[Path]) -> pd.DataFrame:
    rows = [load_run(path) for path in metrics_jsons]                                   # calls load run once per file
    methods = [row["method"] for row in rows]                                           # stacks the rows from load run
    duplicates = sorted({name for name in methods if methods.count(name) > 1})          # keep only the whitelisted metrics
    if duplicates:                                                                      # force values to real numbers
        raise ValueError(f"method names must be unique; duplicates: {duplicates}")      # Hard stop at duplicate names
    table = pd.DataFrame(rows).set_index("method")
    columns = [column for column in COMPARE_METRICS if column in table.columns]
    if not columns:
        raise ValueError("input runs contain no recognized comparison metrics")
    return table[columns].apply(pd.to_numeric, errors="raise")








########    RENDERING    ########

def render_comparison(table: pd.DataFrame, out_html: Path) -> Path:
    figure = plots.comparison_bar_interactive(table, metrics_to_plot=COMPARE_METRICS)           # take table from previous run
    figure_div = figure.to_html(                                                                # draws the grouped bar chart from it
        full_html=False, include_plotlyjs=False, div_id="comparison-bars",                      # formats it into HTML
        config={"displaylogo": False, "responsive": True},
    )
    formatted = table.map(lambda value: f"{value:.4g}")
    table_html = formatted.to_html(border=0, classes="comparison", justify="center", escape=True)
    html = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>DYRK1A method comparison</title><script>{get_plotlyjs()}</script>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1f2937;max-width:1100px;margin:0 auto;padding:28px 20px 60px;background:#f8fafc}}table.comparison{{border-collapse:collapse;width:100%;background:#fff}}table.comparison th,table.comparison td{{padding:9px 14px;border:1px solid #e5e7eb;text-align:center}}.card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:12px;margin:16px 0}}</style></head>
<body><h1>DYRK1A method comparison</h1><p>Each metric has its own vertical scale; EF and AUC are not plotted on one misleading axis.</p><div class='card'>{table_html}</div><div class='card'>{figure_div}</div></body></html>"""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_html.with_name(f".{out_html.name}.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(out_html)
    return out_html






















def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validated analysis runs")
    parser.add_argument("metrics_json", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/comparison.html"))
    args = parser.parse_args()
    table = build_comparison_table(args.metrics_json)
    print(table.to_string())
    print(f"\nWrote comparison to {render_comparison(table, args.out)}")


if __name__ == "__main__":
    main()
