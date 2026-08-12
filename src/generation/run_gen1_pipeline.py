"""Run the frozen GuacaMol Gen1 arm from checkpoint to final report.

The registered campaign is one 10,000-draw generation followed by a uniform
raw 1,000-draw branch through intake, the frozen gate, ligand preparation,
Smina docking, and candidate-only analysis. ``--validation`` permits smaller
counts for a plumbing test and labels the result as a pilot, never as the
registered scientific campaign.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from src.analysis.chemistry import PROPERTY_COLUMNS, PROPERTY_LABELS
from src.analysis.interpretation import campaign_interpretation
from src.analysis.report_theme import report_css, report_toolbar
from src.generation.gen1_analysis import write_gen1_analysis
from src.generation.gen1_guacamol import (
    DECLARED_DOCKING_SUBSAMPLE_SIZE,
    DECLARED_SAMPLE_SIZE,
    DECLARED_SEED,
    MODEL_NAME,
    generate_gen1_samples,
)
from src.generation.run_candidate_pipeline import run_candidate_pipeline
from src.harness import config, intake, runtime


CAMPAIGN_SCHEMA_VERSION = 1
VALIDATION_RAW_DEFAULT = 100
VALIDATION_DOCK_DEFAULT = 10


class Gen1CampaignError(ValueError):
    """Raised before a Gen1 campaign can create ambiguous output."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gen1CampaignError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gen1CampaignError(f"{path} must contain one JSON object")
    return value


def _find_candidate_metrics(candidate_report: Path) -> Path:
    metrics_path = Path(candidate_report).parent / "metrics.json"
    if not metrics_path.is_file():
        raise Gen1CampaignError(
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
    novelty = quality["training_novelty"]
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
    if novelty["available"]:
        novelty_text = (
            f"{novelty['novel_unique_parents']:,} unique parents "
            f"({100 * novelty['novel_unique_parent_fraction']:.2f}%) were absent "
            "from the authenticated GuacaMol training-parent index."
        )
    else:
        novelty_text = str(novelty["reason"])
    source_description = html.escape(candidate.get("source_description", ""))
    run_kind = "Registered campaign" if campaign["design"]["registered_campaign"] else "Pipeline validation pilot"
    neutral_summary = campaign_interpretation(campaign, candidate)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(campaign['name'])}</title>
<style>{report_css()}</style></head>
<body>{report_toolbar("Gen1 campaign")}<h1>Gen1: target-unaware GuacaMol SMILES-LSTM</h1>
<p><strong>{run_kind}.</strong> One frozen checkpoint, one seed, {sampling['raw_samples']:,} raw draws, and one uniformly selected {funnel['submitted']:,}-molecule docking branch. No invalid or duplicate molecule was replaced. The model received no DYRK1A actives, receptor information, docking score, or target reward.</p>
<p><strong>Checkpoint provenance:</strong> {source_description}</p>
<div class="card"><strong>Neutral interpretation</strong><p>{html.escape(neutral_summary)}</p></div>
<h2>1. All generated molecules</h2>
<div class="metrics"><div class="metric">Raw draws<strong>{sampling['raw_samples']:,}</strong></div><div class="metric">Validity<strong>{100 * quality['validity']:.2f}%</strong></div><div class="metric">Parent uniqueness<strong>{100 * quality['uniqueness_among_valid_parents']:.2f}%</strong></div><div class="metric">Accepted / raw<strong>{100 * quality['accepted_fraction']:.2f}%</strong></div><div class="metric">Unique scaffolds<strong>{quality['unique_nonempty_bemis_murcko_scaffolds']:,}</strong></div><div class="metric">Known-active rediscoveries<strong>{quality['exact_known_active_parent_rediscoveries']:,}</strong></div></div>
<div class="card"><p>EOS terminated: {sampling['terminated_by_eos']:,}; hit the 100-character ceiling: {sampling['hit_max_length']:,}; empty raw outputs: {sampling['empty_raw_smiles']:,}.</p><p>{html.escape(novelty_text)}</p></div>
<h3>Full-cohort accepted-parent properties</h3><p>All properties are descriptive. They are neither post-generation filters nor parts of a composite score.</p><div class="card">{property_table}</div>
<h2>2. {funnel['submitted']:,}-molecule screening funnel</h2>
<div class="metrics"><div class="metric">Submitted<strong>{funnel['submitted']:,}</strong></div><div class="metric">Accepted at intake<strong>{funnel['accepted_at_intake']:,}</strong></div><div class="metric">Passed gate<strong>{funnel['passed_predock_gate']:,}</strong></div><div class="metric">Prepared PDBQT<strong>{funnel['prepared_pdbqt_available']:,}</strong></div><div class="metric">Successfully scored<strong>{funnel['successfully_scored']:,}</strong></div><div class="metric">Scored / submitted<strong>{100 * pipeline['rates']['scored_per_submitted']:.2f}%</strong></div></div>
<h3>Observed Smina score distribution</h3><p>Lower is better. Docking is a computational prioritization signal, not proof of biochemical activity. AUC, BEDROC, and enrichment are not computed for this unlabelled candidate cohort.</p><div class="card">{score_table}</div>
<h2>3. Interpretation boundary</h2><p>Gen1 measures the chemical prior learned from the GuacaMol v1 / ChEMBL 24 training universe. It does not test target-conditioned learning and its training corpus is not claimed to be target-disjoint. Any apparent DYRK1A preference is a downstream docking observation and must be interpreted beside validity, uniqueness, survival, molecular-property distributions, and checkpoint provenance.</p>
</body></html>"""


def run_gen1_campaign(
    checkpoint: Path,
    outdir: Path,
    *,
    seed: int = DECLARED_SEED,
    device: str = "cpu",
    workers: int = config.WORKERS,
    actives_path: Path = config.REPO_ROOT
    / "data/reference/dyrk1a_actives_chembl.csv",
    reference_index_dir: Path | None = None,
    raw_n: int = DECLARED_SAMPLE_SIZE,
    dock_n: int = DECLARED_DOCKING_SUBSAMPLE_SIZE,
    validation: bool = False,
) -> Path:
    """Execute the registered design or an explicitly labelled small validation."""

    checkpoint = Path(checkpoint)
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen1CampaignError(f"Gen1 campaign directory already exists: {outdir}")
    if workers < 1:
        raise Gen1CampaignError("workers must be at least 1")
    if raw_n < 1 or dock_n < 1 or dock_n > raw_n:
        raise Gen1CampaignError("Require 1 <= dock_n <= raw_n")
    if not validation and (
        raw_n != DECLARED_SAMPLE_SIZE
        or dock_n != DECLARED_DOCKING_SUBSAMPLE_SIZE
    ):
        raise Gen1CampaignError(
            "Non-registered counts require --validation; the registered campaign "
            "is exactly 10,000 raw draws and 1,000 raw docking submissions"
        )

    samples_dir = outdir / f"samples_{raw_n}"
    sampling = generate_gen1_samples(
        checkpoint,
        samples_dir,
        n=raw_n,
        seed=seed,
        device=device,
    )

    full_intake_dir = outdir / "full_cohort_intake"
    intake.run_intake(samples_dir / "molecules.smi", full_intake_dir)
    full_analysis_dir = outdir / "full_cohort_analysis"
    full_report = write_gen1_analysis(
        samples_dir,
        full_intake_dir,
        actives_path,
        full_analysis_dir,
        reference_index_dir=reference_index_dir,
    )

    checkpoint_hash = sampling["checkpoint"]["weights"]["sha256"]
    source_description = (
        f"{sampling['checkpoint']['declared_source']['description']}; checkpoint "
        f"SHA-256 {checkpoint_hash}; sampling seed {seed}; historical GuacaMol "
        "categorical sampler, batch 64, maximum 100 characters; no "
        "post-generation filtering or top-up"
    )
    suffix = "validation" if validation else "registered"
    name = f"gen1_guacamol_smiles_lstm_v1_{suffix}_{seed}"
    screening_dir = outdir / f"screening_{dock_n}"
    candidate_report = run_candidate_pipeline(
        samples_dir / "molecules.smi",
        screening_dir,
        name=name,
        role="pilot" if validation else "model",
        source_description=source_description,
        n=dock_n,
        seed=seed,
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
        "stage": "gen1_campaign",
        "name": name,
        "model_name": MODEL_NAME,
        "design": {
            "registered_campaign": not validation,
            "independent_model_runs": 1,
            "raw_generated_molecules": raw_n,
            "raw_docking_subsample": dock_n,
            "subsample_before_intake": True,
            "target_aware": False,
            "target_disjoint_training_claimed": False,
            "no_top_up": True,
        },
        "seed": seed,
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
            "Run the official frozen GuacaMol Gen1 checkpoint through generation, "
            "full-cohort audit, and a raw screening branch"
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--seed", type=int, default=DECLARED_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument(
        "--actives",
        type=Path,
        default=config.REPO_ROOT / "data/reference/dyrk1a_actives_chembl.csv",
    )
    parser.add_argument("--reference-index", type=Path)
    parser.add_argument(
        "--validation",
        action="store_true",
        help="label a smaller plumbing run as a pilot, not the registered campaign",
    )
    parser.add_argument("--raw-n", type=int)
    parser.add_argument("--dock-n", type=int)
    args = parser.parse_args()

    raw_n = args.raw_n
    dock_n = args.dock_n
    if raw_n is None:
        raw_n = VALIDATION_RAW_DEFAULT if args.validation else DECLARED_SAMPLE_SIZE
    if dock_n is None:
        dock_n = (
            VALIDATION_DOCK_DEFAULT
            if args.validation
            else DECLARED_DOCKING_SUBSAMPLE_SIZE
        )
    report = run_gen1_campaign(
        args.checkpoint,
        args.outdir,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        actives_path=args.actives,
        reference_index_dir=args.reference_index,
        raw_n=raw_n,
        dock_n=dock_n,
        validation=args.validation,
    )
    print(f"Gen1 campaign report: {report}")


if __name__ == "__main__":
    main()
