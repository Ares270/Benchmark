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
from .interpretation import comparison_interpretation
from .molecule_gallery import molecule_svg
from .report_theme import plotly_config, polish_plotly_figure, report_css, report_toolbar







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


def _ranking_for_run(run: dict) -> pd.DataFrame | None:
    """Load an already written ranking for richer presentation, when present."""

    path = Path(run["_metrics_path"]).parent / "ranked_candidates.csv"
    if not path.is_file():
        return None
    try:
        ranking = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    required = {"molecule_id", "parent_smiles", "score"}
    if not required.issubset(ranking.columns):
        return None
    ranking = ranking.loc[pd.to_numeric(ranking["score"], errors="coerce").notna()].copy()
    ranking["score"] = pd.to_numeric(ranking["score"], errors="coerce")
    return ranking.sort_values(["score", "molecule_id"], kind="stable")


def _funnel_for_run(run: dict) -> dict[str, int]:
    """Recover the durable pipeline funnel; use profile values as a fallback."""

    docking_path = Path(str(run.get("docking", {}).get("path", "")))
    pipeline_path = docking_path.parent.parent / "pipeline_summary.json"
    if pipeline_path.is_file():
        try:
            record = json.loads(pipeline_path.read_text(encoding="utf-8"))
            funnel = record.get("funnel", {})
            return {
                "Submitted": int(funnel["submitted"]),
                "Intake": int(funnel["accepted_at_intake"]),
                "Gate": int(funnel["passed_predock_gate"]),
                "Prepared": int(funnel["prepared_pdbqt_available"]),
                "Scored": int(funnel["successfully_scored"]),
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    intake = run["intake"]
    docking = run["docking"]
    submitted = int(intake["submitted_rows"])
    accepted = int(intake["accepted_for_preparation"])
    scored = int(docking.get("n_with_observed_score", round(
        submitted * float(docking["successful_per_submitted"])
    )))
    return {
        "Submitted": submitted,
        "Intake": accepted,
        "Gate": accepted,
        "Prepared": scored,
        "Scored": scored,
    }


def _arm_cards(summary: pd.DataFrame) -> str:
    cards = []
    for method, row in summary.iterrows():
        role = "Naive baseline" if row["role"] == "naive_baseline" else "Model arm"
        cards.append(
            '<article class="comparison-arm-card">'
            f'<span class="eyebrow">{html.escape(role)}</span>'
            f'<h3>{html.escape(str(method))}</h3>'
            '<div class="comparison-score"><small>Median Smina</small>'
            f'<strong>{row["score_median"]:.3f}</strong><span>kcal/mol</span></div>'
            '<div class="comparison-arm-stats">'
            f'<span><small>Valid</small>{100 * row["validity"]:.1f}%</span>'
            f'<span><small>Unique</small>{100 * row["parent_uniqueness"]:.1f}%</span>'
            f'<span><small>Scored</small>{100 * row["successful_per_submitted"]:.1f}%</span>'
            f'<span><small>CPU-slot h</small>{row["estimated_cpu_slot_hours"]:.2f}</span>'
            '</div></article>'
        )
    return '<div class="comparison-arm-grid">' + "".join(cards) + "</div>"


def _representative_cards(runs: list[dict], rankings: dict[str, pd.DataFrame]) -> str:
    cards = []
    for run in runs:
        ranking = rankings.get(run["name"])
        if ranking is None or ranking.empty:
            continue
        row = ranking.iloc[0]
        cards.append(
            '<article class="molecule-card comparison-molecule">'
            '<div class="molecule-card-head"><span class="molecule-edge best">Top observed</span>'
            f'<span class="molecule-rank">{html.escape(run["name"])}</span></div>'
            f'<div class="molecule-drawing">{molecule_svg(str(row["parent_smiles"]), width=245, height=155)}</div>'
            f'<div class="molecule-id">{html.escape(str(row["molecule_id"]))}</div>'
            f'<div class="molecule-score"><small>Smina</small>{float(row["score"]):.3f} <em>kcal/mol</em></div>'
            '<p class="representative-note">Descriptive extreme only—not an activity claim.</p>'
            '</article>'
        )
    if not cards:
        return ""
    return (
        '<h2>Chemistry quick reference</h2>'
        '<p>One top observed docked structure per arm. Full top-10 and bottom-10 galleries remain in each arm report.</p>'
        '<div class="molecule-grid representative-grid">' + "".join(cards) + "</div>"
    )


def _render_comparison(
    runs: list[dict],
    summary: pd.DataFrame,
    chemistry: pd.DataFrame,
    checks: dict,
) -> str:
    rankings = {
        run["name"]: ranking
        for run in runs
        if (ranking := _ranking_for_run(run)) is not None
    }
    quality = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Generation and screening yield", "Observed docking-score distributions"),
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
    if len(rankings) == len(runs):
        for color, run in zip(("#2563eb", "#0f766e", "#7c3aed", "#d97706"), runs):
            ranking = rankings[run["name"]]
            quality.add_trace(
                go.Box(
                    x=[run["name"]] * len(ranking),
                    y=ranking["score"],
                    name=run["name"],
                    marker_color=color,
                    boxpoints="outliers",
                    showlegend=False,
                    hovertemplate="%{y:.3f} kcal/mol<extra>%{x}</extra>",
                ),
                row=1,
                col=2,
            )
    else:
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
    polish_plotly_figure(quality, height=500)

    funnel_figure = go.Figure()
    for color, run in zip(("#2563eb", "#0f766e", "#7c3aed", "#d97706"), runs):
        funnel = _funnel_for_run(run)
        total = max(1, funnel["Submitted"])
        funnel_figure.add_trace(
            go.Scatter(
                x=list(funnel),
                y=list(funnel.values()),
                customdata=[100 * value / total for value in funnel.values()],
                name=run["name"],
                mode="lines+markers",
                line={"width": 3, "color": color},
                marker={"size": 9},
                hovertemplate="%{y:,} molecules · %{customdata:.1f}% of submitted<extra>%{fullData.name}</extra>",
            )
        )
    funnel_figure.update_yaxes(title_text="Molecules remaining", rangemode="tozero")
    funnel_figure.update_layout(height=430, margin={"t": 35})
    polish_plotly_figure(funnel_figure, height=430)

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
    polish_plotly_figure(property_figure, height=900)

    display_summary = summary.copy()
    for column in ("validity", "parent_uniqueness", "docking_coverage", "successful_per_submitted"):
        display_summary[column] *= 100
    display_summary = display_summary.rename(
        columns={
            "validity": "Validity (%)",
            "parent_uniqueness": "Parent uniqueness (%)",
            "docking_coverage": "Docking coverage",
            "successful_per_submitted": "Scored / submitted (%)",
            "score_p10": "Score p10 (kcal/mol)",
            "score_median": "Score median (kcal/mol)",
            "score_mean": "Score mean (kcal/mol)",
            "estimated_cpu_slot_hours": "Estimated CPU-slot h",
        }
    )
    summary_table = display_summary.to_html(
        border=0,
        classes="data",
        float_format=lambda value: f"{value:.4g}",
    )
    baseline = next(run for run in runs if run["role"] == "naive_baseline")
    toolbar = report_toolbar("Four-arm comparison")
    quality_div = quality.to_html(
        full_html=False, include_plotlyjs=False,
        div_id="candidate-comparison-quality", config=plotly_config(),
    )
    property_div = property_figure.to_html(
        full_html=False, include_plotlyjs=False,
        div_id="candidate-comparison-properties", config=plotly_config(),
    )
    funnel_div = funnel_figure.to_html(
        full_html=False, include_plotlyjs=False,
        div_id="candidate-comparison-funnel", config=plotly_config(),
    )
    neutral = comparison_interpretation(summary.reset_index().to_dict("records"))
    representative_cards = _representative_cards(runs, rankings)
    comparison_css = """
.protocol-banner{display:flex;align-items:center;gap:14px;margin:20px 0;padding:16px 18px;background:var(--teal-soft);border:1px solid #b8dcd6;border-radius:14px}.protocol-banner strong{color:var(--teal-dark)}
.protocol-check{display:grid;place-items:center;flex:0 0 34px;height:34px;color:#fff;background:var(--teal);border-radius:50%;font-size:1.15rem}
.comparison-arm-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:20px 0}.comparison-arm-card{padding:17px;background:#fff;border:1px solid var(--line);border-top:4px solid var(--teal);border-radius:14px;box-shadow:var(--shadow-sm)}.comparison-arm-card:nth-child(2){border-top-color:var(--blue)}.comparison-arm-card:nth-child(3){border-top-color:#7c3aed}.comparison-arm-card:nth-child(4){border-top-color:#d97706}.comparison-arm-card h3{margin:5px 0 13px;overflow-wrap:anywhere}.eyebrow{color:var(--muted);font-size:.68rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}.comparison-score small,.comparison-score span{display:block;color:var(--muted);font-size:.68rem}.comparison-score strong{font-size:1.65rem;line-height:1.1;font-variant-numeric:tabular-nums}.comparison-arm-stats{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:14px}.comparison-arm-stats span{padding:7px;background:var(--canvas);border-radius:8px;font-weight:750;font-variant-numeric:tabular-nums}.comparison-arm-stats small{display:block;color:var(--muted);font-size:.62rem;font-weight:650}.representative-grid{grid-template-columns:repeat(4,minmax(0,1fr))}.comparison-molecule .molecule-rank{max-width:55%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.representative-note{margin:8px 0 0;color:var(--muted);font-size:.72rem}@media(max-width:900px){.comparison-arm-grid,.representative-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){.comparison-arm-grid,.representative-grid{grid-template-columns:1fr}}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DYRK1A candidate comparison</title>
<script>{get_plotlyjs()}</script><style>{report_css()}{comparison_css}</style></head>
<body>{toolbar}<h1>Three model generations versus naïve baseline</h1>
<p>Four-arm outcome review with generation quality, pipeline survival, docking distributions, evaluated-parent chemistry, and measured compute kept visibly separate. There is deliberately no overall winner score.</p>
<div class="protocol-banner"><span class="protocol-check">✓</span><div><strong>Fairness gate passed</strong><br>All four arms submitted {checks['equal_submitted_molecule_budget']:,} raw molecules and used one authenticated receptor and identical scientific docking parameters.</div></div>
<div class="metrics"><div class="metric">Arms compared<strong>4</strong></div><div class="metric">Submitted / arm<strong>{checks['equal_submitted_molecule_budget']:,}</strong></div><div class="metric">Protocol match<strong>Yes</strong></div><div class="metric">Composite score<strong>None</strong></div></div>
{_arm_cards(summary)}
<div class="card"><span class="eyebrow">Neutral interpretation</span><p>{html.escape(neutral)}</p></div>
<div class="card"><strong>Naïve baseline source</strong><p>{html.escape(str(baseline['source_description']))}</p></div>
<h2>Comparable outcomes</h2><p>Percentages use the raw submitted budget as denominator. Score boxes show every observed score when all four ranking files are available; otherwise the report falls back to p10, median, and mean summaries.</p>
<div class="card">{summary_table}</div><div class="plotcard">{quality_div}</div>
<h2>Pipeline survival</h2><p>Absolute molecule counts preserve attrition through intake, the locked pre-dock gate, ligand preparation, and docking.</p><div class="plotcard">{funnel_div}</div>
<h2>Mean evaluated-parent properties</h2><p>Each panel has its own units and scale. These properties are not collapsed into docking affinity.</p>
<div class="plotcard">{property_div}</div>
{representative_cards}
<div class="foot">Docking is a computational prioritization signal, not proof of biochemical activity. Interpret it beside cohort validity, uniqueness, gate survival, molecular properties, and provenance.</div>
</body></html>"""


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
