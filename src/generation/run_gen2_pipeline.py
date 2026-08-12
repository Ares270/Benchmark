"""Run frozen WarmMolGenOne from protein-conditioned generation to report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from src.analysis.chemistry import PROPERTY_COLUMNS, PROPERTY_LABELS
from src.analysis.interpretation import campaign_interpretation
from src.analysis.report_theme import report_css, report_toolbar
from src.generation.gen2_analysis import Gen2AnalysisError, write_gen2_analysis
from src.generation.gen2_warmmolgenone import (
    DEFAULT_CONFIG_PATH,
    MODEL_NAME,
    Gen2CheckpointError,
    Gen2ConfigurationError,
    Gen2SamplingError,
    generate_gen2_samples,
    load_gen2_config,
)
from src.generation.run_candidate_pipeline import (
    CandidatePipelineError,
    run_candidate_pipeline,
)
from src.harness import config, intake, runtime


CAMPAIGN_SCHEMA_VERSION = 1
VALIDATION_RAW_DEFAULT = 100
VALIDATION_DOCK_DEFAULT = 10


class Gen2CampaignError(ValueError):
    """Raised before a Gen2 campaign can create an ambiguous result."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gen2CampaignError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gen2CampaignError(f"{path} must contain one JSON object")
    return value


def _find_candidate_metrics(candidate_report: Path) -> Path:
    metrics_path = Path(candidate_report).parent / "metrics.json"
    if not metrics_path.is_file():
        raise Gen2CampaignError(
            f"Candidate report has no sibling metrics.json: {candidate_report}"
        )
    return metrics_path


def _render_campaign_report(
    campaign: dict,
    full_metrics: dict,
    pipeline: dict,
    candidate: dict,
) -> str:
    quality = full_metrics["generation_quality"]
    sampling = full_metrics["sampling"]
    funnel = pipeline["funnel"]
    scores = candidate["docking"]["score_distribution_kcal_mol"]
    properties = full_metrics["chemistry"]["properties"]
    property_rows = [
        {
            "Property": PROPERTY_LABELS[column],
            "Mean": properties[column]["mean"],
            "Median": properties[column]["median"],
            "SD": properties[column]["std"],
            "Minimum": properties[column]["min"],
            "Maximum": properties[column]["max"],
        }
        for column in PROPERTY_COLUMNS
    ]
    property_table = pd.DataFrame(property_rows).to_html(
        index=False,
        border=0,
        classes="data",
        float_format=lambda value: f"{value:.4g}",
    )
    score_rows = [
        {"Statistic": key, "Value (kcal/mol)": value}
        for key, value in scores.items()
        if key != "n"
    ]
    score_table = pd.DataFrame(score_rows).to_html(
        index=False,
        border=0,
        classes="data",
        float_format=lambda value: f"{value:.4g}",
    )
    run_kind = (
        "Registered campaign"
        if campaign["design"]["registered_campaign"]
        else "Pipeline validation pilot"
    )
    source_description = html.escape(candidate.get("source_description", ""))
    neutral_summary = campaign_interpretation(campaign, candidate)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(campaign['name'])}</title>
<style>{report_css()}</style></head>
<body>{report_toolbar("Gen2 campaign")}<h1>Gen2: target-aware WarmMolGenOne</h1>
<p><strong>{run_kind}.</strong> One frozen checkpoint, one locked DYRK1A kinase-domain sequence, {sampling['raw_samples']:,} raw draws, and one uniformly selected {funnel['submitted']:,}-row screening branch. No failed molecule was replaced.</p>
<p><strong>Provenance:</strong> {source_description}</p>
<div class="card"><strong>Neutral interpretation</strong><p>{html.escape(neutral_summary)}</p></div>
<h2>1. All generated molecules</h2>
<div class="metrics"><div class="metric">Raw draws<strong>{sampling['raw_samples']:,}</strong></div><div class="metric">Validity<strong>{100*quality['validity']:.2f}%</strong></div><div class="metric">Parent uniqueness<strong>{100*quality['uniqueness_among_valid_parents']:.2f}%</strong></div><div class="metric">Accepted / raw<strong>{100*quality['accepted_fraction']:.2f}%</strong></div><div class="metric">Unique scaffolds<strong>{quality['unique_nonempty_bemis_murcko_scaffolds']:,}</strong></div><div class="metric">Known-active rediscoveries<strong>{quality['exact_known_active_parent_rediscoveries']:,}</strong></div></div>
<div class="card"><p>EOS terminated: {sampling['terminated_by_eos']:,}; hit the 128-token ceiling: {sampling['hit_max_length']:,}; empty outputs: {sampling['empty_raw_smiles']:,}.</p><p>{html.escape(quality['training_novelty']['reason'])}</p></div>
<h3>Full-cohort accepted-parent properties</h3><div class="card">{property_table}</div>
<h2>2. Screening funnel</h2>
<div class="metrics"><div class="metric">Submitted<strong>{funnel['submitted']:,}</strong></div><div class="metric">Accepted at intake<strong>{funnel['accepted_at_intake']:,}</strong></div><div class="metric">Passed gate<strong>{funnel['passed_predock_gate']:,}</strong></div><div class="metric">Prepared PDBQT<strong>{funnel['prepared_pdbqt_available']:,}</strong></div><div class="metric">Successfully scored<strong>{funnel['successfully_scored']:,}</strong></div><div class="metric">Scored / submitted<strong>{100*pipeline['rates']['scored_per_submitted']:.2f}%</strong></div></div>
<h3>Observed Smina score distribution</h3><div class="card">{score_table}</div>
<h2>3. Interpretation boundary</h2><p>The model is target-aware because it receives the DYRK1A protein sequence. It is not claimed to be target-disjoint: its frozen weights learned general protein–compound and kinase chemistry from filtered BindingDB interactions. Docking is a prioritization signal, not evidence of biochemical activity.</p>
</body></html>"""


def run_gen2_campaign(
    model_dir: Path,
    decoder_tokenizer_dir: Path,
    outdir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    seed: int | None = None,
    device: str = "cpu",
    workers: int = config.WORKERS,
    actives_path: Path = config.REPO_ROOT
    / "data/reference/dyrk1a_actives_chembl.csv",
    raw_n: int | None = None,
    dock_n: int | None = None,
    validation: bool = False,
) -> Path:
    """Execute the registered Gen2 design or an explicitly labelled pilot."""

    specification = load_gen2_config(config_path)
    declared = specification["sampling"]
    resolved_seed = int(declared["seed"] if seed is None else seed)
    resolved_raw_n = int(declared["raw_samples"] if raw_n is None else raw_n)
    resolved_dock_n = int(
        declared["docking_subsample"] if dock_n is None else dock_n
    )
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen2CampaignError(f"Gen2 campaign directory already exists: {outdir}")
    if workers < 1:
        raise Gen2CampaignError("workers must be at least 1")
    if resolved_raw_n < 1 or not 1 <= resolved_dock_n <= resolved_raw_n:
        raise Gen2CampaignError("Require 1 <= dock_n <= raw_n")
    if not validation and (
        resolved_seed != int(declared["seed"])
        or resolved_raw_n != int(declared["raw_samples"])
        or resolved_dock_n != int(declared["docking_subsample"])
    ):
        raise Gen2CampaignError(
            "Non-registered seed or counts require --validation; the registered "
            "campaign is exactly the locked config design"
        )

    samples_dir = outdir / f"samples_{resolved_raw_n}"
    sampling = generate_gen2_samples(
        model_dir,
        decoder_tokenizer_dir,
        samples_dir,
        config_path=config_path,
        n=resolved_raw_n,
        seed=resolved_seed,
        device=device,
    )
    full_intake_dir = outdir / "full_cohort_intake"
    intake.run_intake(samples_dir / "molecules.smi", full_intake_dir)
    full_analysis_dir = outdir / "full_cohort_analysis"
    full_report = write_gen2_analysis(
        samples_dir,
        full_intake_dir,
        actives_path,
        full_analysis_dir,
    )

    checkpoint_hash = sampling["checkpoint"]["weights"]["sha256"]
    target = sampling["target"]
    source_description = (
        f"Frozen WarmMolGenOne revision {specification['model']['revision']}; "
        f"checkpoint SHA-256 {checkpoint_hash}; conditioned only on human DYRK1A "
        f"Q13627 residues 127-485, target-sequence SHA-256 "
        f"{target['sequence_utf8_lf_sha256']}; seed {resolved_seed}; stochastic "
        "sampling top-k 50, top-p 0.95, temperature 1.0, maximum 128 decoder "
        "tokens; no local training, post-generation filtering, or top-up"
    )
    suffix = "validation" if validation else "registered"
    name = f"gen2_warmmolgenone_v1_{suffix}_{resolved_seed}"
    screening_dir = outdir / f"screening_{resolved_dock_n}"
    candidate_report = run_candidate_pipeline(
        samples_dir / "molecules.smi",
        screening_dir,
        name=name,
        role="pilot" if validation else "model",
        source_description=source_description,
        n=resolved_dock_n,
        seed=resolved_seed,
        workers=workers,
        analysis_outdir=outdir / "candidate_analysis",
    )

    pipeline_summary_path = screening_dir / "pipeline_summary.json"
    candidate_metrics_path = _find_candidate_metrics(candidate_report)
    full_metrics_path = full_analysis_dir / "metrics.json"
    pipeline = _read_json(pipeline_summary_path)
    candidate = _read_json(candidate_metrics_path)
    full_metrics = _read_json(full_metrics_path)
    campaign = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "stage": "gen2_campaign",
        "name": name,
        "model_name": MODEL_NAME,
        "design": {
            "registered_campaign": not validation,
            "independent_model_runs": 1,
            "raw_generated_molecules": resolved_raw_n,
            "raw_docking_subsample": resolved_dock_n,
            "subsample_before_intake": True,
            "target_aware": True,
            "conditioning": "protein sequence only",
            "local_training_or_fine_tuning": False,
            "target_disjoint_training_claimed": False,
            "no_top_up": True,
        },
        "seed": resolved_seed,
        "target": target,
        "checkpoint": sampling["checkpoint"],
        "funnel": pipeline["funnel"],
        "outputs": {
            "full_cohort_report": runtime.file_record(full_report),
            "full_cohort_metrics": runtime.file_record(full_metrics_path),
            "pipeline_summary": runtime.file_record(pipeline_summary_path),
            "candidate_report": runtime.file_record(candidate_report),
            "candidate_metrics": runtime.file_record(candidate_metrics_path),
        },
    }
    runtime.write_json_atomic(outdir / "campaign_summary.json", campaign)
    report_path = outdir / "report.html"
    report_path.write_text(
        _render_campaign_report(campaign, full_metrics, pipeline, candidate),
        encoding="utf-8",
    )
    campaign["outputs"]["combined_report"] = runtime.file_record(report_path)
    runtime.write_json_atomic(outdir / "campaign_summary.json", campaign)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen WarmMolGenOne through generation, full-cohort audit, "
            "and a raw screening branch"
        )
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("decoder_tokenizer_dir", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument(
        "--actives",
        type=Path,
        default=config.REPO_ROOT / "data/reference/dyrk1a_actives_chembl.csv",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="label smaller counts as a pilot rather than the registered campaign",
    )
    parser.add_argument("--raw-n", type=int)
    parser.add_argument("--dock-n", type=int)
    args = parser.parse_args()

    raw_n = args.raw_n
    dock_n = args.dock_n
    if args.validation:
        if raw_n is None:
            raw_n = VALIDATION_RAW_DEFAULT
        if dock_n is None:
            dock_n = VALIDATION_DOCK_DEFAULT
    try:
        report = run_gen2_campaign(
            args.model_dir,
            args.decoder_tokenizer_dir,
            args.outdir,
            config_path=args.config,
            seed=args.seed,
            device=args.device,
            workers=args.workers,
            actives_path=args.actives,
            raw_n=raw_n,
            dock_n=dock_n,
            validation=args.validation,
        )
    except (
        Gen2AnalysisError,
        Gen2CampaignError,
        Gen2CheckpointError,
        Gen2ConfigurationError,
        Gen2SamplingError,
        CandidatePipelineError,
        FileNotFoundError,
    ) as error:
        parser.error(str(error))
    print(f"Gen2 campaign report: {report}")


if __name__ == "__main__":
    main()
