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
from .chemistry import PROPERTY_COLUMNS, PROPERTY_LABELS
from .report_theme import plotly_config, polish_plotly_figure, report_css, report_toolbar

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





######## load property means from multiple runs ########

def build_chemistry_comparison_table(               
    metrics_jsons: list[Path],
) -> pd.DataFrame | None:
    """Load active/candidate parent-property means from compatible run files."""

    rows = []
    missing = []
    for metrics_json in metrics_jsons:                          # For every metrics.json, the script extracts mean properties from the active/candidate cohort
        path = Path(metrics_json)                               # If no selected runs have chemistry, the old docking-only comparison still works
        data = json.loads(path.read_text(encoding="utf-8"))     # If some runs have chemistry and some do not, it refuses the comparison
        method = str(data.get("name") or path.parent.name)
        chemical = data.get("chemistry")
        if chemical is None:
            missing.append(method)
            continue
        try:
            properties = chemical["cohorts"]["actives"]["properties"]
            row = {
                "method": method,
                **{
                    column: properties[column]["mean"]
                    for column in PROPERTY_COLUMNS
                },
            }
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"{path} has an incomplete chemistry profile"
            ) from error
        rows.append(row)

    if not rows:
        return None
    if missing:
        raise ValueError(
            "chemistry comparison requires chemistry profiles for every run; "
            f"missing: {missing}"
        )
    methods = [row["method"] for row in rows]
    duplicates = sorted({name for name in methods if methods.count(name) > 1})
    if duplicates:
        raise ValueError(f"method names must be unique; duplicates: {duplicates}")
    table = pd.DataFrame(rows).set_index("method")
    return table[list(PROPERTY_COLUMNS)].apply(pd.to_numeric, errors="raise")








########    RENDERING    ########

def render_comparison(
    table: pd.DataFrame,
    out_html: Path,
    chemistry_table: pd.DataFrame | None = None,
) -> Path:
    figure = plots.comparison_bar_interactive(table, metrics_to_plot=COMPARE_METRICS)           # take table from previous run
    polish_plotly_figure(figure)
    figure_div = figure.to_html(                                                                # draws the grouped bar chart from it
        full_html=False, include_plotlyjs=False, div_id="comparison-bars",                      # formats it into HTML
        config=plotly_config(),
    )
    formatted = table.map(lambda value: f"{value:.4g}")
    table_html = formatted.to_html(border=0, classes="comparison", justify="center", escape=True)
    chemistry_section = ""
    if chemistry_table is not None:
        chemistry_figure = plots.chemical_mean_comparison_interactive(
            chemistry_table
        )
        polish_plotly_figure(chemistry_figure)
        chemistry_figure_div = chemistry_figure.to_html(            ##### NEW CHEMISTRY PLOT
            full_html=False,                                            # When chemistry exists, it additionally draws:
            include_plotlyjs=False,                                     # -A table of mean properties
            div_id="chemistry-comparison-bars",                         # -Twelve separate bar-chart panels
            config=plotly_config(),
        )
        chemistry_formatted = chemistry_table.rename(
            columns=PROPERTY_LABELS
        ).map(lambda value: f"{value:.4g}")
        chemistry_table_html = chemistry_formatted.to_html(
            border=0, classes="comparison", justify="center", escape=True
        )
        chemistry_section = (
            "<h2>Mean evaluated-parent properties</h2>"
            "<p>These are cohort means, not a composite quality score. "
            "Each panel keeps the property’s own units and scale.</p>"
            f"<div class='card'>{chemistry_table_html}</div>"
            f"<div class='card'>{chemistry_figure_div}</div>"
        )
    toolbar = report_toolbar("Labelled method comparison")
    html = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>DYRK1A method comparison</title><script>{get_plotlyjs()}</script>
<style>{report_css()}</style></head>
<body>{toolbar}<h1>DYRK1A method comparison</h1><p>Each metric has its own vertical scale; EF and AUC are not plotted on one misleading axis.</p><div class='card'>{table_html}</div><div class='plotcard'>{figure_div}</div>{chemistry_section}<div class='foot'>Every metric remains separate. This report does not calculate an overall winner score.</div></body></html>"""
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
    chemistry_table = build_chemistry_comparison_table(args.metrics_json)
    print(table.to_string())
    if chemistry_table is not None:
        print("\nMean evaluated-parent properties:\n" + chemistry_table.to_string())
    print(
        f"\nWrote comparison to {render_comparison(table, args.out, chemistry_table)}"
    )


if __name__ == "__main__":
    main()
