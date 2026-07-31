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

from . import chemistry, config, metrics, plots, provenance, report, size as size_module
from .dataset import AnalysisInputError, build_dataset




#####  Replace, Indivisible Process, Failsafe for half finished reports  #####

def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")                     # Temp file path matters
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)




####  No Mor NaN in JSON files  ####

def _json_text(value: dict) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"          # Fail Loud 



def _load_docking_cost(scores_path: Path) -> tuple[dict | None, Path | None]:           # new    
    """Load a sibling docking summary only when it matches this score file."""          # looks beside a score file for
                                                                                        # _dock_summary.json
    summary_path = Path(scores_path).parent / "_dock_summary.json"
    if not summary_path.is_file():
        return None, None
    try:
        record = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise AnalysisInputError(
            f"Cannot read docking cost summary {summary_path}: {error}"
        ) from error
    if record.get("stage") != "smina_docking":
        raise AnalysisInputError(f"{summary_path} is not a Smina docking summary")
    recorded_hash = (
        record.get("outputs", {})
        .get("scores_csv", {})
        .get("sha256")
    )
    actual_hash = provenance.file_record(scores_path)["sha256"]
    if not recorded_hash or recorded_hash != actual_hash:
        raise AnalysisInputError(
            f"{summary_path} does not match the analyzed scores file {scores_path}"
        )
    return record, summary_path



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
    chemical_frame: pd.DataFrame | None = None,     # new chemical figures
    chemical_profile: dict | None = None,           # if the chemistry analysis is done
    size_frame: pd.DataFrame | None = None,         # heavy-atom diagnostic
) -> None:
    factories = {
        "roc_curve": lambda: plots.roc_static(fpr, tpr, auc),
        "score_distribution": lambda: plots.score_distribution_static(active_observed, decoy_observed, score_direction),
        "enrichment_curve": lambda: plots.enrichment_static(frac_screened, frac_found, ef_points),
        "rank_plot": lambda: plots.rank_static(frame, score_direction),
        "violin_plot": lambda: plots.violin_static(active_observed, decoy_observed),
    }
    if chemical_frame is not None and chemical_profile is not None:
        factories.update(
            {
                "chemical_distributions": lambda: plots.chemical_distributions_static(
                    chemical_frame
                ),
                "chemical_landscape": lambda: plots.chemical_landscape_static(chemical_frame),
                "score_property_correlations": lambda: plots.chemical_correlations_static(
                    chemical_profile
                ),
            }
        )
    if size_frame is not None:
        factories["size_dependence"] = lambda: plots.size_dependence_static(
            size_frame, score_direction
        )
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
    active_intake_path: Path | None = None,     # this change was made to allow for chemical profiling
    decoy_intake_path: Path | None = None,
    active_smi_path: Path | None = None,        # docked .smi inputs, for the size diagnostic
    decoy_smi_path: Path | None = None,
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
    if (active_intake_path is None) != (decoy_intake_path is None):
        raise AnalysisInputError(
            "Chemical profiling requires both active and decoy intake molecules.csv files"
        )
    if (active_smi_path is None) != (decoy_smi_path is None):
        raise AnalysisInputError(
            "The size diagnostic requires both the active and the decoy .smi input"
        )

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
    active_cost, active_cost_path = _load_docking_cost(scores_path)     ##### Lines 247–254 load active and decoy cost records when available
    decoy_cost, decoy_cost_path = _load_docking_cost(decoy_path)
    computational_cost = None
    if active_cost is not None or decoy_cost is not None:
        computational_cost = {
            "actives": active_cost,
            "decoys": decoy_cost,
        }
    report.check_harness_configuration(computational_cost)      # cohorts docked under
                                                                # different settings are not
                                                                # comparable; fail before any
                                                                # output directory exists
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

    chemical_frame = None
    chemical_profile = None
    if active_intake_path is not None and decoy_intake_path is not None:        # If both intake files are present, 
        chemical_frame, chemical_profile = chemistry.build_chemical_profile(    # the chemistry module validates and summarizes them
            active_intake_path, decoy_intake_path, frame
        )

    size_frame = None
    size_profile = None
    if active_smi_path is not None and decoy_smi_path is not None:              # heavy-atom count for every analyzed molecule
        size_frame, size_profile = size_module.build_size_profile(              # raises if any scored ID has no input SMILES
            frame, active_smi_path, decoy_smi_path
        )

    run_dir, final_dir, timestamp = _run_directories(outdir, name)
    figure_dir = run_dir / config.FIG_SUBDIR
    interactive_dir = run_dir / config.INTERACTIVE_SUBDIR
    figure_dir.mkdir()
    interactive_dir.mkdir()

    _save_static_figures(
        figure_dir, frame, active_observed, decoy_observed,
        fpr, tpr, auc, frac_screened, frac_found, ef_points, score_direction,
        chemical_frame=chemical_frame,
        chemical_profile=chemical_profile,
        size_frame=size_frame,
    )
    interactive_figures = {
        "roc": plots.roc_interactive(fpr, tpr, auc),
        "score_distribution": plots.score_distribution_interactive(active_observed, decoy_observed, score_direction),
        "enrichment": plots.enrichment_interactive(frac_screened, frac_found, ef_points),
        "rank": plots.rank_interactive(frame, score_direction),
        "violin": plots.violin_interactive(active_observed, decoy_observed),
    }
    if chemical_frame is not None and chemical_profile is not None:
        interactive_figures.update(
            {
                "chemical_distributions": plots.chemical_distributions_interactive(chemical_frame),
                "chemical_landscape": plots.chemical_landscape_interactive(chemical_frame),
                "score_property_correlations": plots.chemical_correlations_interactive(chemical_profile),
            }
        )
    if size_frame is not None:
        interactive_figures["size_dependence"] = plots.size_dependence_interactive(
            size_frame, score_direction
        )
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
    if active_intake_path is not None and decoy_intake_path is not None:
        input_records["active_intake"] = provenance.file_record(active_intake_path)
        input_records["decoy_intake"] = provenance.file_record(decoy_intake_path)
        input_records["active_intake_summary"] = provenance.file_record(active_intake_path.parent / "summary.json")
        input_records["decoy_intake_summary"] = provenance.file_record(decoy_intake_path.parent / "summary.json")
    if active_smi_path is not None and decoy_smi_path is not None:
        input_records["active_smi"] = provenance.file_record(active_smi_path)
        input_records["decoy_smi"] = provenance.file_record(decoy_smi_path)
    if active_cost_path is not None:
        input_records["active_docking_summary"] = provenance.file_record(active_cost_path)
    if decoy_cost_path is not None:
        input_records["decoy_docking_summary"] = provenance.file_record(decoy_cost_path)
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
        chemistry=chemical_profile,
        computational_cost=computational_cost,
        size=size_profile,
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
    if chemical_profile is not None:
        metrics_output["chemistry"] = chemical_profile
    if size_profile is not None:
        metrics_output["size_dependence"] = size_profile
    if computational_cost is not None:
        metrics_output["computational_cost"] = computational_cost
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
            "chemical_profile_enabled": chemical_profile is not None,
            "size_diagnostic_enabled": size_profile is not None,
        },
        "computational_cost": computational_cost,
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
    parser.add_argument(
        "--active-intake",
        type=Path,
        default=None,
        help="optional active intake molecules.csv; requires --decoy-intake",
    )
    parser.add_argument(
        "--decoy-intake",
        type=Path,
        default=None,
        help="optional decoy intake molecules.csv; requires --active-intake",
    )
    parser.add_argument(
        "--active-smi",
        type=Path,
        default=None,
        help="actives .smi that was docked; enables the size diagnostic, requires --decoy-smi",
    )
    parser.add_argument(
        "--decoy-smi",
        type=Path,
        default=None,
        help="decoys .smi that was docked; requires --active-smi",
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
            active_intake_path=args.active_intake,
            decoy_intake_path=args.decoy_intake,
            active_smi_path=args.active_smi,
            decoy_smi_path=args.decoy_smi,
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
