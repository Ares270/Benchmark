"""Run frozen pocket-conditioned Molexar from generation to docking report."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.analysis.chemistry import PROPERTY_COLUMNS, PROPERTY_LABELS
from src.analysis.interpretation import campaign_interpretation
from src.analysis.report_theme import report_css, report_toolbar
from src.generation.gen3_analysis import Gen3AnalysisError, write_gen3_analysis
from src.generation.gen3_molexar import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SOURCE_ROOT,
    MODEL_NAME,
    Gen3ConfigurationError,
    load_gen3_config,
)
from src.generation.run_candidate_pipeline import (
    CandidatePipelineError,
    run_candidate_pipeline,
)
from src.harness import config, intake, runtime


CAMPAIGN_SCHEMA_VERSION = 1
VALIDATION_RAW_DEFAULT = 100
VALIDATION_DOCK_DEFAULT = 10
DEFAULT_MOLEXAR_PYTHON = (
    config.REPO_ROOT / "Models & Miscellaneous/molexar-env/bin/python"
)


class Gen3CampaignError(ValueError):
    """Raised before a Gen3 campaign can create an ambiguous result."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gen3CampaignError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gen3CampaignError(f"{path} must contain one JSON object")
    return value


def _find_candidate_metrics(candidate_report: Path) -> Path:
    metrics_path = Path(candidate_report).parent / "metrics.json"
    if not metrics_path.is_file():
        raise Gen3CampaignError(
            f"Candidate report has no sibling metrics.json: {candidate_report}"
        )
    return metrics_path












#########    Generation does not run inside dyrk1a-bench. 
#########    It runs in a completely separate Python interpreter with its own package versions, 
#########    launched as a child process.



def _run_generation_subprocess(
    molexar_python: Path,
    model_dir: Path,
    target_pdb: Path,
    source_root: Path,
    config_path: Path,
    sample_dir: Path,
    *,
    n: int,
    seed: int,
    device: str,
) -> dict:
    molexar_python = Path(molexar_python)
    if not molexar_python.is_file():
        raise Gen3CampaignError(
            f"Isolated Molexar Python executable does not exist: {molexar_python}"
        )
    command = [
        str(molexar_python.resolve()),
        "-m",
        "src.generation.gen3_molexar",
        str(Path(model_dir).resolve()),
        str(Path(target_pdb).resolve()),
        str(Path(sample_dir).resolve()),
        "--source-root",
        str(Path(source_root).resolve()),
        "--config",
        str(Path(config_path).resolve()),
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    try:
        completed = subprocess.run(command, cwd=config.REPO_ROOT, check=False)
    except OSError as error:
        raise Gen3CampaignError(f"Could not start isolated Molexar runtime: {error}") from error
    if completed.returncode != 0:
        raise Gen3CampaignError(
            f"Isolated Molexar generation exited with code {completed.returncode}"
        )
    sampling_path = Path(sample_dir) / "sampling.json"
    sampling = _read_json(sampling_path)
    if sampling.get("stage") != "gen3_sampling":
        raise Gen3CampaignError(
            f"Isolated runtime did not produce a Gen3 sampling record: {sampling_path}"
        )
    if int(sampling.get("counts", {}).get("raw_samples", -1)) != n:
        raise Gen3CampaignError("Isolated runtime returned the wrong raw sample count")
    return sampling


def _render_campaign_report(
    campaign: dict,
    full_metrics: dict,
    pipeline: dict,
    candidate: dict,
) -> str:
    quality = full_metrics["generation_quality"]
    sampling = full_metrics["sampling"]
    pocket = sampling["pocket_graph"]
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
<body>{report_toolbar("Gen3 campaign")}<h1>Gen3: pocket-aware Molexar</h1>
<p><strong>{run_kind}.</strong> One frozen checkpoint, one locked 7O7K protein pocket, {sampling['raw_samples']:,} raw draws, and one uniformly selected {funnel['submitted']:,}-row screening branch. No failed molecule was replaced.</p>
<p><strong>Provenance:</strong> {source_description}</p>
<div class="card"><strong>Neutral interpretation</strong><p>{html.escape(neutral_summary)}</p></div>
<h2>1. All generated molecules</h2>
<div class="metrics"><div class="metric">Raw draws<strong>{sampling['raw_samples']:,}</strong></div><div class="metric">Converted<strong>{sampling['fragment_selfies_conversion_successes']:,}</strong></div><div class="metric">Validity<strong>{100*quality['validity']:.2f}%</strong></div><div class="metric">Parent uniqueness<strong>{100*quality['uniqueness_among_valid_parents']:.2f}%</strong></div><div class="metric">Accepted / raw<strong>{100*quality['accepted_fraction']:.2f}%</strong></div><div class="metric">Unique scaffolds<strong>{quality['unique_nonempty_bemis_murcko_scaffolds']:,}</strong></div><div class="metric">Known-active rediscoveries<strong>{quality['exact_known_active_parent_rediscoveries']:,}</strong></div></div>
<div class="card"><p>Pocket graph: {pocket['atoms_used']} atoms, {pocket['edge_count']} directed edges, SHA-256 {pocket['graph_sha256']}. Fragment-SELFIES conversion failures: {sampling['fragment_selfies_conversion_failures']:,}. No invalid or duplicate output was replaced.</p><p>{html.escape(quality['training_novelty']['reason'])}</p></div>
<h3>Full-cohort accepted-parent properties</h3><div class="card">{property_table}</div>
<h2>2. Screening funnel</h2>
<div class="metrics"><div class="metric">Submitted<strong>{funnel['submitted']:,}</strong></div><div class="metric">Accepted at intake<strong>{funnel['accepted_at_intake']:,}</strong></div><div class="metric">Passed gate<strong>{funnel['passed_predock_gate']:,}</strong></div><div class="metric">Prepared PDBQT<strong>{funnel['prepared_pdbqt_available']:,}</strong></div><div class="metric">Successfully scored<strong>{funnel['successfully_scored']:,}</strong></div><div class="metric">Scored / submitted<strong>{100*pipeline['rates']['scored_per_submitted']:.2f}%</strong></div></div>
<h3>Observed Smina score distribution</h3><div class="card">{score_table}</div>
<h2>3. Interpretation boundary</h2><p>The model is target-aware because it receives the protein-pocket atom graph. It receives no active ligand, pharmacophore, molecular-property target, sequence embedding, gate result, Smina score, or optimization reward. Its training is not claimed to be target-disjoint. Docking is a prioritization signal, not evidence of biochemical activity.</p>
</body></html>"""


def run_gen3_campaign(
    model_dir: Path,
    outdir: Path,
    *,
    molexar_python: Path = DEFAULT_MOLEXAR_PYTHON,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    target_pdb: Path | None = None,
    seed: int | None = None,
    device: str = "cpu",
    workers: int = config.WORKERS,
    actives_path: Path = config.REPO_ROOT
    / "data/reference/dyrk1a_actives_chembl.csv",
    raw_n: int | None = None,
    dock_n: int | None = None,
    validation: bool = False,
) -> Path:
    """Execute the registered Gen3 design or an explicitly labelled pilot."""

    specification = load_gen3_config(config_path)
    declared = specification["sampling"]
    resolved_seed = int(declared["seed"] if seed is None else seed)
    resolved_raw_n = int(declared["raw_samples"] if raw_n is None else raw_n)
    resolved_dock_n = int(
        declared["docking_subsample"] if dock_n is None else dock_n
    )
    resolved_target = (
        config.REPO_ROOT / specification["target"]["source_file"]
        if target_pdb is None
        else Path(target_pdb)
    )
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen3CampaignError(f"Gen3 campaign directory already exists: {outdir}")
    if workers < 1:
        raise Gen3CampaignError("workers must be at least 1")
    if resolved_raw_n < 1 or not 1 <= resolved_dock_n <= resolved_raw_n:
        raise Gen3CampaignError("Require 1 <= dock_n <= raw_n")
    
    
    

# btw you cannot run a "real" campaign with anything other than the locked config values

    if not validation and (
        resolved_seed != int(declared["seed"])
        or resolved_raw_n != int(declared["raw_samples"])
        or resolved_dock_n != int(declared["docking_subsample"])
    ):
        raise Gen3CampaignError(
            "Non-registered seed or counts require --validation; the registered "
            "campaign is exactly the locked config design"
        )









    samples_dir = outdir / f"samples_{resolved_raw_n}"
    sampling = _run_generation_subprocess(
        molexar_python,
        model_dir,
        resolved_target,
        source_root,
        config_path,
        samples_dir,
        n=resolved_raw_n,
        seed=resolved_seed,
        device=device,
    )
    full_intake_dir = outdir / "full_cohort_intake"
    intake.run_intake(samples_dir / "molecules.smi", full_intake_dir)
    full_analysis_dir = outdir / "full_cohort_analysis"
    full_report = write_gen3_analysis(
        samples_dir,
        full_intake_dir,
        actives_path,
        full_analysis_dir,
    )

    checkpoint_hash = sampling["checkpoint"]["weights"]["sha256"]
    pocket = sampling["pocket_graph"]
    source_description = (
        f"Frozen fairydance/molexar-10m-omni revision "
        f"{specification['model']['revision']}; checkpoint SHA-256 "
        f"{checkpoint_hash}; conditioned only on the 7O7K protein pocket centered "
        f"at {specification['target']['center_angstrom']} Angstrom, radius "
        f"{specification['target']['pocket_radius_angstrom']} Angstrom, "
        f"{pocket['atoms_used']} atoms, pocket-graph SHA-256 "
        f"{pocket['graph_sha256']}; seed {resolved_seed}; stochastic sampling "
        "temperature 0.8, top-k 50, top-p 0.95, maximum 204 new tokens; no "
        "local training, active-ligand input, docking reward, filtering, or top-up"
    )
    suffix = "validation" if validation else "registered"
    name = f"gen3_molexar_pocket_v1_{suffix}_{resolved_seed}"
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
        "stage": "gen3_campaign",
        "name": name,
        "model_name": MODEL_NAME,
        "design": {
            "registered_campaign": not validation,
            "independent_model_runs": 1,
            "raw_generated_molecules": resolved_raw_n,
            "raw_docking_subsample": resolved_dock_n,
            "subsample_before_intake": True,
            "target_aware": True,
            "conditioning": "protein-pocket coordinates only",
            "local_training_or_fine_tuning": False,
            "target_disjoint_training_claimed": False,
            "active_ligand_or_docking_reward_used": False,
            "no_top_up": True,
        },
        "seed": resolved_seed,
        "target": sampling["target"],
        "pocket_graph": pocket,
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
            "Run frozen pocket-conditioned Molexar through generation, "
            "full-cohort audit, and a raw screening branch"
        )
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--molexar-python", type=Path, default=DEFAULT_MOLEXAR_PYTHON)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--target-pdb", type=Path)
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
        help="label smaller counts as a pilot rather than registered campaign",
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
        report = run_gen3_campaign(
            args.model_dir,
            args.outdir,
            molexar_python=args.molexar_python,
            source_root=args.source_root,
            config_path=args.config,
            target_pdb=args.target_pdb,
            seed=args.seed,
            device=args.device,
            workers=args.workers,
            actives_path=args.actives,
            raw_n=raw_n,
            dock_n=dock_n,
            validation=args.validation,
        )
    except (
        CandidatePipelineError,
        FileNotFoundError,
        Gen3AnalysisError,
        Gen3CampaignError,
        Gen3ConfigurationError,
    ) as error:
        parser.error(str(error))
    print(f"Gen3 campaign report: {report}")


if __name__ == "__main__":
    main()
