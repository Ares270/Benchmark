"""Compare exactly three model candidate runs and one naive baseline.

The comparison is budget- and protocol-gated.  It keeps generation quality,
docking scores, molecular properties, and computational cost on separate axes;
it does not manufacture a composite winner score.


it is compare.py for the unlabeled half of the benchmark, with fairness enforcement replacing the metric whitelist.

"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
from plotly import graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from src.harness import runtime

from .chemistry import PROPERTY_COLUMNS, PROPERTY_LABELS







################     GATEKEEPER SCRIPT    #################


PROTOCOL_KEYS = (
    "smina_cpu_per_job",        # these parameters must match across all runs
    "box_center_a",
    "box_size_a",
    "exhaustiveness",
    "seed",
    "num_modes",
    "energy_range_kcal_mol",
    "timeout_seconds_per_ligand",
)

                                                                                    # CRITERIA TO EVEN WRITE ANYTHING
class CandidateComparisonError(ValueError):
    """Raised when candidate runs are not scientifically comparable."""                 # Four runs exactly
                                                                                        # 3 models and one baseline exactly
                                                                                        # identical submitted budget
def _load_run(path: Path) -> dict:                                                      # byte-identical receptor and docking parameters
    path = Path(path)                                                                   # authenticated cost data on every run
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateComparisonError(f"Cannot read {path}: {error}") from error
    if run.get("stage") != "candidate_analysis":
        raise CandidateComparisonError(f"{path} is not a candidate-analysis metrics file")  # Inputs are 4 metrics.json files
    run["_metrics_path"] = str(path.resolve())
    return run


def _scientific_protocol(run: dict) -> dict:
    protocol = run.get("screening_protocol")
    if not isinstance(protocol, dict):
        raise CandidateComparisonError(
            f"{run.get('name')} has no authenticated docking protocol"
        )
    receptor = protocol.get("receptor")
    if not isinstance(receptor, dict) or not receptor.get("sha256"):
        raise CandidateComparisonError(
            f"{run.get('name')} has no hashed receptor identity"
        )
    parameters = protocol.get("parameters")
    if not isinstance(parameters, dict):
        raise CandidateComparisonError(
            f"{run.get('name')} has no docking parameters"
        )
    missing = [key for key in PROTOCOL_KEYS if key not in parameters]
    if missing:
        raise CandidateComparisonError(
            f"{run.get('name')} docking protocol lacks {missing}"
        )
    return {
        "receptor_sha256": receptor["sha256"],
        "parameters": {key: parameters[key] for key in PROTOCOL_KEYS},
    }










def build_candidate_comparison(
    metrics_paths: list[Path],
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict]:
    """Validate the four-run design and return summary/property tables."""                    # Load and base check each file
                                                                                              # Names non blank and unique
    if len(metrics_paths) != 4:                                                               # Role census (3 models 1 baseline)
        raise CandidateComparisonError(                                                       # baseline has a source description
            "Candidate comparison requires exactly four runs: three models and one naive baseline"
        )                                                                                     # submitted budgets identical
    runs = [_load_run(path) for path in metrics_paths]                                        # protocols identical
    names = [str(run.get("name", "")).strip() for run in runs]                                # per-run cost exists
    if any(not name for name in names):                                                       # assemble two DataFrames
        raise CandidateComparisonError("Every candidate run needs a nonblank name")             
    duplicates = sorted({name for name in names if names.count(name) > 1})                    # then....
    if duplicates:
        raise CandidateComparisonError(f"Candidate run names must be unique: {duplicates}")   # _render_comparison builds the HTML
                                                                                              # write_candidate_comparison writes to disk
    roles = [run.get("role") for run in runs]
    if roles.count("model") != 3 or roles.count("naive_baseline") != 1:
        raise CandidateComparisonError(
            "Roles must be exactly three 'model' runs and one 'naive_baseline' run"
        )
    baseline = runs[roles.index("naive_baseline")]
    if not str(baseline.get("source_description", "")).strip():
        raise CandidateComparisonError(
            "Naive baseline needs a source_description stating its sampling universe and rule"
        )

    submitted = [int(run["intake"]["submitted_rows"]) for run in runs]
    if len(set(submitted)) != 1:
        raise CandidateComparisonError(
            "Submitted-molecule budgets differ across runs: "
            + ", ".join(f"{name}={count}" for name, count in zip(names, submitted))
        )

    protocols = [_scientific_protocol(run) for run in runs]
    protocol_text = [json.dumps(value, sort_keys=True) for value in protocols]
    if len(set(protocol_text)) != 1:
        raise CandidateComparisonError(
            "Receptor or scientific docking parameters differ across candidate runs"
        )

    summary_rows = []
    chemistry_rows = []
    for run in runs:
        intake = run["intake"]
        docking = run["docking"]
        scores = docking["score_distribution_kcal_mol"]
        cost = run.get("computational_cost")
        if not isinstance(cost, dict):
            raise CandidateComparisonError(
                f"{run['name']} has no authenticated computational-cost summary"
            )
        summary_rows.append(
            {
                "method": run["name"],
                "role": run["role"],
                "submitted": int(intake["submitted_rows"]),
                "validity": float(intake["validity"]),
                "parent_uniqueness": float(intake["parent_uniqueness"]),
                "accepted": int(intake["accepted_for_preparation"]),
                "docking_coverage": float(docking["coverage_over_accepted_parents"]),
                "successful_per_submitted": float(docking["successful_per_submitted"]),
                "score_p10": float(scores["p10"]),
                "score_median": float(scores["median"]),
                "score_mean": float(scores["mean"]),
                "estimated_cpu_slot_hours": float(
                    cost.get("timing", {}).get(
                        "estimated_requested_cpu_slot_hours",
                        0.0,
                    )
                ),
            }
        )
        properties = run["chemistry"]["properties"]
        chemistry_rows.append(
            {
                "method": run["name"],
                **{
                    column: float(properties[column]["mean"])
                    for column in PROPERTY_COLUMNS
                },
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("method")
    chemistry = pd.DataFrame(chemistry_rows).set_index("method")
    checks = {
        "exactly_four_runs": True,
        "three_models_one_naive_baseline": True,
        "equal_submitted_molecule_budget": submitted[0],
        "identical_receptor_and_scientific_docking_parameters": True,
        "scientific_protocol": protocols[0],
        "no_composite_winner_score": True,
    }
    return runs, summary, chemistry, checks


def _render_comparison(
    runs: list[dict],
    summary: pd.DataFrame,
    chemistry: pd.DataFrame,
    checks: dict,
) -> str:
    quality = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Generation and screening yield", "Docking-score distribution summaries"),
    )
    for column, label in (
        ("validity", "Validity"),
        ("parent_uniqueness", "Parent uniqueness"),
        ("successful_per_submitted", "Scored / submitted"),
    ):
        quality.add_trace(
            go.Bar(x=summary.index, y=100 * summary[column], name=label),
            row=1,
            col=1,
        )
    for column, label in (
        ("score_p10", "10th percentile"),
        ("score_median", "Median"),
        ("score_mean", "Mean"),
    ):
        quality.add_trace(
            go.Bar(x=summary.index, y=summary[column], name=label),
            row=1,
            col=2,
        )
    quality.update_yaxes(title_text="Percent of submitted", row=1, col=1)
    quality.update_yaxes(title_text="kcal/mol (lower is better)", row=1, col=2)
    quality.update_layout(height=480, barmode="group", margin={"t": 60})

    property_figure = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=[PROPERTY_LABELS[column] for column in PROPERTY_COLUMNS],
    )
    for position, column in enumerate(PROPERTY_COLUMNS):
        property_figure.add_trace(
            go.Bar(
                x=chemistry.index,
                y=chemistry[column],
                showlegend=False,
                marker_color=["#2563eb", "#0f766e", "#7c3aed", "#d97706"],
            ),
            row=position // 4 + 1,
            col=position % 4 + 1,
        )
    property_figure.update_layout(height=900, margin={"t": 60})

    display_summary = summary.rename(
        columns={
            "validity": "Validity",
            "parent_uniqueness": "Parent uniqueness",
            "docking_coverage": "Docking coverage",
            "successful_per_submitted": "Scored / submitted",
            "score_p10": "Score p10",
            "score_median": "Score median",
            "score_mean": "Score mean",
            "estimated_cpu_slot_hours": "Estimated CPU-slot h",
        }
    )
    summary_table = display_summary.to_html(
        border=0,
        classes="data",
        float_format=lambda value: f"{value:.4g}",
    )
    baseline = next(run for run in runs if run["role"] == "naive_baseline")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DYRK1A candidate comparison</title><script>{get_plotlyjs()}</script><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1250px;margin:auto;padding:28px 20px 60px;background:#f8fafc;color:#1f2937}}.card{{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:16px 0}}table.data{{border-collapse:collapse;width:100%}}table.data th,table.data td{{border:1px solid #e5e7eb;padding:7px;text-align:right}}table.data th:first-child,table.data td:first-child{{text-align:left}}</style></head><body><h1>Three model generations versus naïve baseline</h1><p>All four runs submitted <strong>{checks['equal_submitted_molecule_budget']:,}</strong> raw molecules and used one identical receptor and scientific docking protocol. Invalids, duplicates, and docking failures remain visible. There is deliberately no overall winner score.</p><p><strong>Naïve baseline:</strong> {html.escape(str(baseline['source_description']))}</p><div class="card">{summary_table}</div><div class="card">{quality.to_html(full_html=False,include_plotlyjs=False,div_id='candidate-comparison-quality',config={'displaylogo':False,'responsive':True})}</div><h2>Mean evaluated-parent properties</h2><p>Each panel has its own units and scale. These properties are not collapsed into docking affinity.</p><div class="card">{property_figure.to_html(full_html=False,include_plotlyjs=False,div_id='candidate-comparison-properties',config={'displaylogo':False,'responsive':True})}</div></body></html>"""


def write_candidate_comparison(
    metrics_paths: list[Path],
    outdir: Path,
) -> Path:
    runs, summary, chemistry, checks = build_candidate_comparison(metrics_paths)
    outdir = Path(outdir)
    if outdir.exists():
        raise CandidateComparisonError(f"Comparison output path already exists: {outdir}")
    outdir.mkdir(parents=True)
    summary.to_csv(outdir / "summary.csv")
    chemistry.to_csv(outdir / "chemistry_means.csv")
    (outdir / "report.html").write_text(
        _render_comparison(runs, summary, chemistry, checks),
        encoding="utf-8",
    )
    runtime.write_json_atomic(
        outdir / "comparison.json",
        {
            "schema_version": 1,
            "stage": "four_run_candidate_comparison",
            "design_checks": checks,
            "runs": [
                {
                    "name": run["name"],
                    "role": run["role"],
                    "metrics_path": run["_metrics_path"],
                    "source_description": run.get("source_description", ""),
                }
                for run in runs
            ],
            "outputs": {
                "summary_csv": runtime.file_record(outdir / "summary.csv"),
                "chemistry_means_csv": runtime.file_record(outdir / "chemistry_means.csv"),
                "report_html": runtime.file_record(outdir / "report.html"),
            },
        },
    )
    return outdir / "report.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three model candidate runs with one naive baseline"
    )
    parser.add_argument("metrics_json", type=Path, nargs=4)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = write_candidate_comparison(args.metrics_json, args.outdir)
    except CandidateComparisonError as error:
        parser.error(str(error))
    print(f"Candidate comparison: {report}")


if __name__ == "__main__":
    main()
