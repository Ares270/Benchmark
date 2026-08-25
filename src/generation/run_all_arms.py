"""Run the four registered candidate arms and fairness-gated comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.compare_candidates import (
    CandidateComparisonError,
    write_candidate_comparison,
)
from src.generation.run_candidate_pipeline import (
    CandidatePipelineError,
    run_candidate_pipeline,
)
from src.generation.run_gen1_pipeline import Gen1CampaignError, run_gen1_campaign
from src.generation.run_gen2_pipeline import Gen2CampaignError, run_gen2_campaign
from src.generation.run_gen3_pipeline import (
    DEFAULT_MOLEXAR_PYTHON,
    Gen3CampaignError,
    run_gen3_campaign,
)
from src.harness import config, runtime


ALL_ARMS_SCHEMA_VERSION = 1
REGISTERED_SEED = 20260801
REGISTERED_RAW_N = 10000
REGISTERED_DOCK_N = 1000

DEFAULT_BASELINE = (
    config.REPO_ROOT
    / "data/generated/naive_property_matched_20260801/molecules.smi"
)
DEFAULT_GEN1_CHECKPOINT = (
    config.REPO_ROOT / "Models & Miscellaneous/model_final_0.473.pt"
)
DEFAULT_GEN2_MODEL = config.REPO_ROOT / "Models & Miscellaneous"
DEFAULT_GEN2_TOKENIZER = (
    config.REPO_ROOT / "Models & Miscellaneous/PubChem10M_SMILES_BPE_450k"
)
DEFAULT_GEN3_MODEL = (
    config.REPO_ROOT / "Models & Miscellaneous/molexar-10m-omni"
)


class AllArmsCampaignError(ValueError):
    """Raised before or during the immutable registered four-arm campaign."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AllArmsCampaignError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AllArmsCampaignError(f"{path} must contain one JSON object")
    return value


def _campaign_candidate_metrics(campaign_dir: Path) -> Path:
    summary_path = Path(campaign_dir) / "campaign_summary.json"
    summary = _read_json(summary_path)
    path = Path(summary.get("outputs", {}).get("candidate_metrics", {}).get("path", ""))
    if not path.is_file():
        raise AllArmsCampaignError(
            f"Campaign has no candidate metrics output: {summary_path}"
        )
    return path





#### Orchestrator Functions ####

def run_all_arms(
    outdir: Path,
    *,
    workers: int = config.WORKERS,
    baseline_smi: Path = DEFAULT_BASELINE,
    gen1_checkpoint: Path = DEFAULT_GEN1_CHECKPOINT,
    gen2_model_dir: Path = DEFAULT_GEN2_MODEL,
    gen2_tokenizer_dir: Path = DEFAULT_GEN2_TOKENIZER,
    gen3_model_dir: Path = DEFAULT_GEN3_MODEL,
    molexar_python: Path = DEFAULT_MOLEXAR_PYTHON,
) -> Path:
    """Execute exactly one registered run per arm, then compare all four."""

    outdir = Path(outdir)
    if outdir.exists():
        raise AllArmsCampaignError(f"All-arm output directory already exists: {outdir}")
    if workers < 1:
        raise AllArmsCampaignError("workers must be at least 1")
    required_inputs = {
        "registered naive baseline": Path(baseline_smi),
        "Gen1 checkpoint": Path(gen1_checkpoint),
        "Gen2 model weights": Path(gen2_model_dir) / "pytorch_model.bin",
        "Gen2 decoder tokenizer": Path(gen2_tokenizer_dir) / "vocab.json",
        "Gen3 model weights": Path(gen3_model_dir) / "pytorch_model.bin",
        "Gen3 isolated Python": Path(molexar_python),
    }
    missing = [f"{label}: {path}" for label, path in required_inputs.items() if not path.is_file()]
    if missing:
        raise AllArmsCampaignError("Required all-arm inputs are missing: " + "; ".join(missing))

    outdir.mkdir(parents=True)
    baseline_dir = outdir / "naive_property_matched"
    baseline_report = run_candidate_pipeline(
        baseline_smi,
        baseline_dir / "screening_1000",
        name="naive_property_matched_v1_20260801",
        role="naive_baseline",
        source_description=(
            "Property-matched naive baseline; ChEMBL 37 parent structures; "
            "known-active parents excluded by RDKit-computed parent InChIKey; "
            "generator seed 20260801; uniform raw 1000-row screening branch; "
            "no post-generation top-up"
        ),
        n=REGISTERED_DOCK_N,
        seed=REGISTERED_SEED,
        workers=workers,
        analysis_outdir=baseline_dir / "candidate_analysis",
    )
    baseline_metrics = baseline_report.parent / "metrics.json"

    gen1_dir = outdir / "gen1_guacamol"
    run_gen1_campaign(
        gen1_checkpoint,
        gen1_dir,
        seed=REGISTERED_SEED,
        device="cpu",
        workers=workers,
        raw_n=REGISTERED_RAW_N,
        dock_n=REGISTERED_DOCK_N,
        validation=False,
    )
    gen1_metrics = _campaign_candidate_metrics(gen1_dir)

    gen2_dir = outdir / "gen2_warmmolgenone"
    run_gen2_campaign(
        gen2_model_dir,
        gen2_tokenizer_dir,
        gen2_dir,
        seed=REGISTERED_SEED,
        device="cpu",
        workers=workers,
        raw_n=REGISTERED_RAW_N,
        dock_n=REGISTERED_DOCK_N,
        validation=False,
    )
    gen2_metrics = _campaign_candidate_metrics(gen2_dir)

    gen3_dir = outdir / "gen3_molexar"
    run_gen3_campaign(
        gen3_model_dir,
        gen3_dir,
        molexar_python=molexar_python,
        seed=REGISTERED_SEED,
        device="cpu",
        workers=workers,
        raw_n=REGISTERED_RAW_N,
        dock_n=REGISTERED_DOCK_N,
        validation=False,
    )
    gen3_metrics = _campaign_candidate_metrics(gen3_dir)

    metrics_paths = [baseline_metrics, gen1_metrics, gen2_metrics, gen3_metrics]
    comparison_report = write_candidate_comparison(
        metrics_paths,
        outdir / "four_arm_comparison",
    )
    runtime.write_json_atomic(
        outdir / "all_arms_summary.json",
        {
            "schema_version": ALL_ARMS_SCHEMA_VERSION,
            "stage": "registered_all_arms_campaign",
            "seed": REGISTERED_SEED,
            "raw_samples_per_model": REGISTERED_RAW_N,
            "submitted_per_arm": REGISTERED_DOCK_N,
            "workers": workers,
            "arms": {
                "naive_property_matched": runtime.file_record(baseline_metrics),
                "gen1_guacamol": runtime.file_record(gen1_metrics),
                "gen2_warmmolgenone": runtime.file_record(gen2_metrics),
                "gen3_molexar": runtime.file_record(gen3_metrics),
            },
            "comparison_report": runtime.file_record(comparison_report),
        },
    )
    return comparison_report
















def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the registered naive baseline, Gen1, Gen2, and Gen3, then "
            "apply the four-way candidate-comparison fairness gate"
        )
    )
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--gen1-checkpoint", type=Path, default=DEFAULT_GEN1_CHECKPOINT)
    parser.add_argument("--gen2-model-dir", type=Path, default=DEFAULT_GEN2_MODEL)
    parser.add_argument("--gen2-tokenizer-dir", type=Path, default=DEFAULT_GEN2_TOKENIZER)
    parser.add_argument("--gen3-model-dir", type=Path, default=DEFAULT_GEN3_MODEL)
    parser.add_argument("--molexar-python", type=Path, default=DEFAULT_MOLEXAR_PYTHON)
    args = parser.parse_args()
    try:
        report = run_all_arms(
            args.outdir,
            workers=args.workers,
            baseline_smi=args.baseline,
            gen1_checkpoint=args.gen1_checkpoint,
            gen2_model_dir=args.gen2_model_dir,
            gen2_tokenizer_dir=args.gen2_tokenizer_dir,
            gen3_model_dir=args.gen3_model_dir,
            molexar_python=args.molexar_python,
        )
    except (
        AllArmsCampaignError,
        CandidateComparisonError,
        CandidatePipelineError,
        FileNotFoundError,
        Gen1CampaignError,
        Gen2CampaignError,
        Gen3CampaignError,
    ) as error:
        parser.error(str(error))
    print(f"Registered four-arm comparison report: {report}")


if __name__ == "__main__":
    main()
