"""
Docking scores CSV (actives) + Decoys scores CSV -> a full evaluation 
in a timestamped run directory.

    python -m src.analysis.run_analysis \
        --scores       results/docking/actives_scores.csv \
        --decoy-scores results/docking/decoys_scores.csv \
        --labels       data/reference/dyrk1a_actives_chembl.csv \
        --name harness_validation_v1 --outdir results/


Actives come from --scores (label 1), 
Decoys from --decoy-scores (label 0).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config, metrics, plots, provenance, report
from .dataset import AnalysisInputError, build_dataset




#####  Replace, Indivisible Process, Failsafe for half finished reports  #####

def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")                     # Temp file path matters
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)




####  No Mor NaN in JSON files  ####

def _json_text(value: dict) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"          # Fail Loud 




####  Input Sanitization, with Allow list  ####

def _run_directories(outdir: Path, name: str) -> tuple[Path, Path, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
        raise AnalysisInputError(
            "Run name must be 1-80 characters using only letters, digits, '.', '_', or '-'"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    final_dir = Path(outdir) / f"{name}_{timestamp}"
    working_dir = final_dir.with_name(f".{final_dir.name}.partial")
    if final_dir.exists() or working_dir.exists():
        raise AnalysisInputError(f"Output run path already exists: {final_dir}")
    working_dir.mkdir(parents=True)
    return working_dir, final_dir, timestamp                                # Atomic flip





#############    Metrics    ################

def _save_static_figures(
    fig_dir: Path,
    frame: pd.DataFrame,
    active_observed,
    decoy_observed,
    fpr,
    tpr,
    auc: float,
    frac_screened,
    frac_found,
    ef_points,
    score_direction: str,
) -> None:
    factories = {
        "roc_curve": lambda: plots.roc_static(fpr, tpr, auc),
        "score_distribution": lambda: plots.score_distribution_static(active_observed, decoy_observed, score_direction),
        "enrichment_curve": lambda: plots.enrichment_static(frac_screened, frac_found, ef_points),
        "rank_plot": lambda: plots.rank_static(frame, score_direction),
        "violin_plot": lambda: plots.violin_static(active_observed, decoy_observed),
    }
    for filename, factory in factories.items():
        figure = factory()
        try:
            figure.savefig(fig_dir / f"{filename}.png")
            figure.savefig(fig_dir / f"{filename}.svg")
        finally:
            plt.close(figure)






#######  History  #######

def _append_history(
    outdir: Path,
    timestamp: str,
    name: str,
    audit: dict,
    metric_values: dict,
    run_dir: Path,
) -> None:
    history_path = Path(outdir) / config.RUN_HISTORY_NAME
    row = {
        "timestamp_utc": timestamp,
        "name": name,
        "n_actives": audit["n_analyzed_actives"],
        "n_decoys": audit["n_analyzed_decoys"],
        **metric_values,
        "outdir": str(run_dir.resolve()),
    }
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = history.rename(columns={"timestamp": "timestamp_utc", "ef1": "ef_1pct", "ef5": "ef_5pct"})
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True, sort=False)
    else:
        history = pd.DataFrame([row])
    temporary = history_path.with_name(f".{history_path.name}.tmp")
    history.to_csv(temporary, index=False)
    temporary.replace(history_path)






###########     Summary     ###########

def _print_summary(name: str, audit: dict, metric_values: dict, intervals: dict, report_path: Path) -> None:
    print(f"\nDYRK1A benchmark — {name}")
    print(f"  analyzed: {audit['n_analyzed_actives']:,} actives | {audit['n_analyzed_decoys']:,} decoys")
    for key, value in metric_values.items():
        interval = intervals.get(key)
        suffix = f"  [{interval['low']:.3f}, {interval['high']:.3f}]" if interval else ""
        print(f"  {key}: {value:.4f}{suffix}")
    print(f"  report: {report_path}")
















#############  Run  #############

def run(
    scores_path: Path,
    decoy_path: Path,
    labels_path: Path | None,
    name: str,
    outdir: Path,
    *,
    missing_policy: str = config.MISSING_SCORE_POLICY,
    score_direction: str = config.SCORE_DIRECTION,              #Score direction comes from a single source 
    bootstrap_replicates: int = config.BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = config.BOOTSTRAP_SEED,
    confidence_level: float = config.CONFIDENCE_LEVEL,
    id_column: str = config.ID_COLUMN,
    score_column: str = config.SCORE_COLUMN,
    reference_id_column: str = config.REFERENCE_ID_COLUMN,
) -> Path:
    """Validate inputs, calculate metrics, and write one immutable run directory."""

    started = time.monotonic()
    dataset = build_dataset(
        scores_path,
        decoy_path,
        labels_path,
        missing_policy=missing_policy,
        score_direction=score_direction,
        id_column=id_column,
        score_column=score_column,
        reference_id_column=reference_id_column,
    )
    frame = dataset.frame
    labels = frame["label"].to_numpy(int)
    scores = frame["score"].to_numpy(float)

    metric_values = metrics.summary_metrics(
        labels,
        scores,
        ef_fractions=config.EF_FRACTIONS,
        alpha=config.BEDROC_ALPHA,
        score_direction=score_direction,
    )
    fpr, tpr, auc = metrics.roc_curve(labels, scores, score_direction)
    frac_screened, frac_found = metrics.enrichment_curve(labels, scores, score_direction)
    ef_points = [
        (fraction, *metrics.enrichment_operating_point(labels, scores, fraction, score_direction))
        for fraction in config.EF_FRACTIONS
    ]
                                                                                        # Mark rows where docks failed 
    observed = frame.loc[~frame["score_imputed"]]                                       # Select non imputed ones         
    active_observed = observed.loc[observed["label"] == 1, "score"].to_numpy(float)     # For no Mann-Whitney interference
    decoy_observed = observed.loc[observed["label"] == 0, "score"].to_numpy(float)      # Even though we keep imputed values 
    statistics = metrics.score_statistics(                                              # for enrichment tests
        active_observed,                                                                # Planned asymmetry
        decoy_observed,
        alternative=config.MANNWHITNEY_ALTERNATIVE,
        score_direction=score_direction,
    )
    intervals = metrics.bootstrap_confidence_intervals(
        labels,
        scores,
        n_resamples=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        ef_fractions=config.EF_FRACTIONS,
        alpha=config.BEDROC_ALPHA,
        score_direction=score_direction,
    )

    run_dir, final_dir, timestamp = _run_directories(outdir, name)
    figure_dir = run_dir / config.FIG_SUBDIR
    interactive_dir = run_dir / config.INTERACTIVE_SUBDIR
    figure_dir.mkdir()
    interactive_dir.mkdir()

    _save_static_figures(
        figure_dir, frame, active_observed, decoy_observed,
        fpr, tpr, auc, frac_screened, frac_found, ef_points, score_direction,
    )
    interactive_figures = {
        "roc": plots.roc_interactive(fpr, tpr, auc),
        "score_distribution": plots.score_distribution_interactive(active_observed, decoy_observed, score_direction),
        "enrichment": plots.enrichment_interactive(frac_screened, frac_found, ef_points),
        "rank": plots.rank_interactive(frame, score_direction),
        "violin": plots.violin_interactive(active_observed, decoy_observed),
    }
    for filename, figure in interactive_figures.items():
        figure.write_html(
            interactive_dir / f"{filename}.html",
            include_plotlyjs=True,
            full_html=True,
            config={"displaylogo": False, "responsive": True},
        )

    repo_root = Path(__file__).resolve().parents[2]
    versions = provenance.software_versions()
    input_records = {
        "active_scores": provenance.file_record(scores_path),
        "decoy_scores": provenance.file_record(decoy_path),
    }
    if labels_path is not None:
        input_records["active_reference"] = provenance.file_record(labels_path)
    provenance_record = {
        "inputs": input_records,
        "git": provenance.git_state(repo_root),
        "harness": provenance.harness_snapshot(),
    }
    timestamp_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "name": name,
        "timestamp_iso": timestamp_iso,
        "wall_seconds": time.monotonic() - started,
        "confidence_level": confidence_level,
    }

    html = report.render_report(
        meta,
        metric_values,
        intervals,
        statistics,
        interactive_figures,
        versions,
        dataset.audit,
        provenance_record,
    )
    report_path = run_dir / config.REPORT_NAME
    _write_text_atomic(report_path, html)

    metrics_output = {
        "schema_version": config.OUTPUT_SCHEMA_VERSION,                     # Version for the JSON files
        "name": name,
        "dataset": {
            "n_actives": dataset.audit["n_analyzed_actives"],
            "n_decoys": dataset.audit["n_analyzed_decoys"],
            "missing_policy": missing_policy,
        },
        "metrics": metric_values,
        "confidence_intervals": intervals,
        "bootstrap": {
            "method": "class-stratified compound-level percentile",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "score_statistics": statistics,
        "ef_fractions": list(config.EF_FRACTIONS),
        "bedroc_alpha": config.BEDROC_ALPHA,
        "score_direction": score_direction,
    }
    _write_text_atomic(run_dir / config.METRICS_NAME, _json_text(metrics_output))

    run_log = {
        "schema_version": config.OUTPUT_SCHEMA_VERSION,
        **meta,
        "command": sys.argv,                                # Command for the Run
        "parameters": {
            "id_column": id_column,
            "score_column": score_column,
            "reference_id_column": reference_id_column,
            "score_direction": score_direction,
            "missing_policy": missing_policy,
            "ef_fractions": list(config.EF_FRACTIONS),
            "bedroc_alpha": config.BEDROC_ALPHA,
            "mannwhitney_alternative": config.MANNWHITNEY_ALTERNATIVE,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "dataset_audit": dataset.audit,
        "provenance": provenance_record,
        "versions": versions,
        "outdir": str(final_dir.resolve()),
    }
    _write_text_atomic(run_dir / config.RUN_LOG_NAME, _json_text(run_log))

    run_dir.replace(final_dir)
    report_path = final_dir / config.REPORT_NAME
    _append_history(outdir, timestamp, name, dataset.audit, metric_values, final_dir)
    _print_summary(name, dataset.audit, metric_values, intervals, report_path)
    return report_path



























#########  Command Line  #########

def main() -> None:
    parser = argparse.ArgumentParser(description="Validated DYRK1A docking ranking analysis")
    parser.add_argument("--scores", type=Path, required=True, help="harness CSV for known actives")
    parser.add_argument("--decoy-scores", type=Path, required=True, help="harness CSV for benchmark decoys")
    parser.add_argument(
        "--active-reference", "--labels", dest="labels", type=Path, default=None,
        help="optional active-reference CSV used as a strict identity check",
    )
    parser.add_argument("--name", required=True, help="safe run identifier used in output paths")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--id-column", default=config.ID_COLUMN)
    parser.add_argument("--score-column", default=config.SCORE_COLUMN)
    parser.add_argument("--reference-id-column", default=config.REFERENCE_ID_COLUMN)
    parser.add_argument(
        "--score-direction", choices=config.VALID_SCORE_DIRECTIONS,
        default=config.SCORE_DIRECTION,
    )
    parser.add_argument(
        "--missing-policy", choices=config.VALID_MISSING_SCORE_POLICIES,
        default=config.MISSING_SCORE_POLICY,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=config.BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=config.BOOTSTRAP_SEED)
    parser.add_argument("--confidence-level", type=float, default=config.CONFIDENCE_LEVEL)
    args = parser.parse_args()
    try:
        run(
            args.scores,
            args.decoy_scores,
            args.labels,
            args.name,
            args.outdir,
            missing_policy=args.missing_policy,
            score_direction=args.score_direction,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
            id_column=args.id_column,
            score_column=args.score_column,
            reference_id_column=args.reference_id_column,
        )
    except AnalysisInputError as error:                                             # Cleaner analysis errors
        parser.error(str(error))


if __name__ == "__main__":
    main()
