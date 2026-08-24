"""Sample the frozen pocket-conditioned Molexar Gen3 checkpoint.

Every model draw is retained. Fragment-SELFIES decoding failures are transported
as deliberately invalid placeholders so intake, rather than the generator,
decides validity. No ligand, property, sequence, gate, or docking-score signal
is supplied to the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.harness import config, runtime


GEN3_SCHEMA_VERSION = 1
MODEL_NAME = "gen3_molexar_pocket_v1"
DEFAULT_CONFIG_PATH = config.REPO_ROOT / "configs/gen3_molexar.json"
DEFAULT_SOURCE_ROOT = config.REPO_ROOT / "Models & Miscellaneous/arm3_sources"
RAW_SAMPLES_NAME = "raw_samples.csv"
MOLECULES_NAME = "molecules.smi"
SAMPLING_NAME = "sampling.json"
EMPTY_TRANSPORT_TOKEN = "__GEN3_EMPTY_SMILES__"
WHITESPACE_TRANSPORT_TOKEN = "__GEN3_WHITESPACE_SMILES__"


class Gen3ConfigurationError(ValueError):
    """Raised when the locked Gen3 scientific configuration is inconsistent."""


class Gen3CheckpointError(ValueError):
    """Raised when the local model/runtime is not the registered Arm 3 bundle."""


class Gen3SamplingError(ValueError):
    """Raised when pocket construction or raw generation violates the contract."""


def load_gen3_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read and validate the versioned Gen3 scientific configuration."""






##### alor of the following code is from the original source and should be preserved as is   ########


    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gen3ConfigurationError(f"Cannot read Gen3 config {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gen3ConfigurationError("Gen3 config must contain one JSON object")
    if value.get("schema_version") != GEN3_SCHEMA_VERSION:
        raise Gen3ConfigurationError("Unsupported Gen3 config schema version")
    arm = value.get("arm", {})
    if arm.get("name") != MODEL_NAME:
        raise Gen3ConfigurationError("Gen3 config arm name does not match implementation")
    if arm.get("conditioning") != "protein_pocket_coordinates":
        raise Gen3ConfigurationError("Gen3 must remain pocket-coordinate conditioned")
    if arm.get("local_training_or_fine_tuning"):
        raise Gen3ConfigurationError("Gen3 local training/fine-tuning must remain off")            # guardrails for cheating

    target = value.get("target", {})
    center = tuple(float(item) for item in target.get("center_angstrom", ()))
    if center != tuple(float(item) for item in config.BOX_CENTER):
        raise Gen3ConfigurationError(
            "Gen3 pocket center must equal the locked docking center"
        )
    if float(target.get("pocket_radius_angstrom", -1)) <= 0:
        raise Gen3ConfigurationError("Gen3 pocket radius must be positive")
    if int(target.get("max_atoms", -1)) < 1:
        raise Gen3ConfigurationError("Gen3 pocket atom limit must be positive")
    if target.get("include_hydrogens") is not False:
        raise Gen3ConfigurationError("Gen3 pocket hydrogens must remain excluded")

    sampling = value.get("sampling", {})
    for field in ("seed", "raw_samples", "docking_subsample", "batch_size"):
        if int(sampling.get(field, -1)) < 1:
            raise Gen3ConfigurationError(f"Gen3 {field} must be positive")
    if int(sampling["docking_subsample"]) > int(sampling["raw_samples"]):
        raise Gen3ConfigurationError("Gen3 docking subsample cannot exceed raw samples")
    
    
    
    # even more guardrails for cheating
    
    if sampling.get("do_sample") is not True:
        raise Gen3ConfigurationError("Gen3 stochastic sampling must remain enabled")
    if sampling.get("validity_filtering") or sampling.get("deduplication"):
        raise Gen3ConfigurationError("Gen3 generator filtering/deduplication must remain off")
    if sampling.get("replacement_or_top_up"):
        raise Gen3ConfigurationError("Gen3 replacement/top-up must remain off")
    
    
    
    if not sampling.get("conversion_canonical"):
        raise Gen3ConfigurationError("Gen3 Fragment-SELFIES conversion must remain canonical")
    if sampling.get("conversion_randomized") or sampling.get("conversion_strict"):
        raise Gen3ConfigurationError(
            "Gen3 conversion must remain non-randomized and non-strict"
        )
    return value


def _verify_artifact(path: Path, expected: dict[str, Any], label: str) -> dict:
    if not path.is_file():
        raise Gen3CheckpointError(f"{label} does not exist: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != int(expected["bytes"]):
        raise Gen3CheckpointError(
            f"{label} byte count mismatch: expected {expected['bytes']}, "
            f"observed {observed_bytes}"
        )
    observed_hash = runtime.sha256_file(path)
    if observed_hash != expected["sha256"]:
        raise Gen3CheckpointError(
            f"{label} SHA-256 mismatch: expected {expected['sha256']}, "
            f"observed {observed_hash}"
        )
    return runtime.file_record(path)


def verify_checkpoint(model_dir: Path, specification: dict[str, Any]) -> dict[str, Any]:
    """Authenticate every official model artifact before loading weights."""

    model_dir = Path(model_dir)
    records = {
        name: _verify_artifact(model_dir / name, expected, f"Molexar {name}")
        for name, expected in specification["model"]["artifacts"].items()
    }
    return {
        "declared_model": specification["model"],
        "artifacts": records,
        "weights": records["pytorch_model.bin"],
        "local_model_dir": str(model_dir.resolve()),
    }


def verify_source_tree(source_root: Path, specification: dict[str, Any]) -> dict:
    """Authenticate the exact pinned Molexar and Fragment-SELFIES Python sources."""

    source_root = Path(source_root)
    roots = (source_root / "Molexar", source_root / "Fragment-SELFIES")
    paths = sorted(path for root in roots for path in root.rglob("*.py"))
    observed = runtime.file_set_record(paths)
    expected = specification["official_sources"]["python_source_set"]
    if observed != expected:
        raise Gen3CheckpointError(
            f"Arm 3 official source fingerprint mismatch: expected {expected}, "
            f"observed {observed}"
        )
    return {
        **observed,
        "root": str(source_root.resolve()),
        "revisions": {
            key: value["revision"]
            for key, value in specification["official_sources"].items()
            if isinstance(value, dict) and "revision" in value
        },
    }


def verify_runtime_versions(specification: dict[str, Any]) -> dict[str, str]:
    """Refuse an unregistered Molexar software stack."""

    observed = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "transformers": _installed_version("transformers"),
        "tokenizers": _installed_version("tokenizers"),
        "rdkit": _installed_version("rdkit"),
        "numpy": np.__version__,
        "scipy": _installed_version("scipy"),
        "fragment_selfies": _installed_version("fragment-selfies"),
        "molexar": _installed_version("molexar"),
    }
    expected = {key: str(value) for key, value in specification["runtime"].items()}
    if observed != expected:
        raise Gen3CheckpointError(
            f"Arm 3 runtime mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _installed_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError as error:
        raise Gen3CheckpointError(f"Arm 3 runtime package is missing: {name}") from error


def verify_target(target_pdb: Path, specification: dict[str, Any]) -> dict:
    target_pdb = Path(target_pdb)
    expected = specification["target"]
    record = _verify_artifact(
        target_pdb,
        {"bytes": expected["source_bytes"], "sha256": expected["source_sha256"]},
        "Gen3 target PDB",
    )
    return record


def load_model_bundle(
    model_dir: Path,
    source_root: Path,
    *,
    device: str,
    specification: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Load the authenticated local Molexar bundle without network fallback."""

    runtime_versions = verify_runtime_versions(specification)
    checkpoint = verify_checkpoint(model_dir, specification)
    sources = verify_source_tree(source_root, specification)
    try:
        from molexar.inference import MolexarInference

        engine = MolexarInference(str(model_dir), device=device)
    except Exception as error:
        raise Gen3CheckpointError(f"Cannot load local Molexar checkpoint: {error}") from error
    observed = {
        "parameters": sum(parameter.numel() for parameter in engine.model.parameters()),
        "model_type": str(engine.config.model_type),
        "vocab_size": int(engine.config.vocab_size),
        "hidden_size": int(engine.config.hidden_size),
        "num_hidden_layers": int(engine.config.num_hidden_layers),
        "gvp_node_in_dim": list(engine.config.gvp_node_in_dim),
        "gvp_edge_in_dim": list(engine.config.gvp_edge_in_dim),
    }
    expected = {
        "parameters": int(specification["model"]["expected_parameters"]),
        "model_type": "molexar",
        "vocab_size": 127,
        "hidden_size": 256,
        "num_hidden_layers": 16,
        "gvp_node_in_dim": [11, 3],
        "gvp_edge_in_dim": [32, 1],
    }
    if observed != expected:
        raise Gen3CheckpointError(
            f"Molexar architecture mismatch: expected {expected}, observed {observed}"
        )
    checkpoint["observed_architecture"] = observed
    checkpoint["official_sources"] = sources
    checkpoint["runtime"] = runtime_versions
    return engine, checkpoint


def _graph_fingerprint(graph: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(graph):
        value = graph[key]
        digest.update(key.encode("utf-8") + b"\0")
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()





#### the only chemistry input ####

def build_pocket_graph(
    engine: Any,
    target_pdb: Path,
    specification: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the pinned official processor around the fixed docking center."""

    target = specification["target"]
    try:
        from molexar.datasets.pocket import PocketProcessor

        processor = PocketProcessor(
            pocket_radius=float(target["pocket_radius_angstrom"]),
            max_atoms=int(target["max_atoms"]),
            include_hydrogens=False,
            centroid=tuple(float(item) for item in target["center_angstrom"]),
            node_scalar_dim=int(engine.config.gvp_node_in_dim[0]),
        )
        all_coords, atoms = processor._parse_pdb(str(target_pdb))
        pocket_coords, _ = processor._extract_pocket(
            all_coords, atoms, processor.centroid
        )
        graph = processor.process_pdb(str(target_pdb))
    except Exception as error:
        raise Gen3SamplingError(f"Cannot construct the registered pocket graph: {error}") from error
    observed_counts = {
        "all_nonhydrogen_atoms": len(all_coords),
        "atoms_within_radius_before_truncation": len(pocket_coords),
        "atoms_used": int(graph["n_atoms"]),
    }
    expected_counts = {
        key: int(target[key])
        for key in (
            "all_nonhydrogen_atoms",
            "atoms_within_radius_before_truncation",
            "atoms_used",
        )
    }
    if observed_counts != expected_counts:
        raise Gen3SamplingError(
            f"Registered pocket atom counts changed: expected {expected_counts}, "
            f"observed {observed_counts}"
        )
    record = {
        **observed_counts,
        "center_angstrom": [float(item) for item in graph["center"]],
        "radius_angstrom": float(target["pocket_radius_angstrom"]),
        "max_atoms": int(target["max_atoms"]),
        "knn_k": int(target["knn_k"]),
        "edge_count": int(graph["edge_index"].shape[1]),
        "graph_sha256": _graph_fingerprint(graph),
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in graph.items()
            if isinstance(value, torch.Tensor)
        },
    }
    return graph, record













######    The DRAW    #######

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_molecules(
    engine: Any,
    pocket_graph: dict[str, Any],
    *,
    n: int,
    seed: int,
    sampling: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exactly N raw Fragment-SELFIES draws and conversions."""

    if n < 1:
        raise Gen3SamplingError("n must be at least 1")
    if seed < 0:
        raise Gen3SamplingError("seed must be non-negative")
    _seed_everything(seed)
    try:
        fragment_rows = engine.generate(
            conditions={"prot_poc_gvp_emb": pocket_graph},
            max_new_tokens=int(sampling["max_new_tokens"]),
            num_samples=n,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
            do_sample=True,
            repetition_penalty=float(sampling["repetition_penalty"]),
            batch_size=int(sampling["batch_size"]),
        )
        converted = engine.convert_to_smiles(
            fragment_rows,
            canonical=True,
            randomized=False,
            strict=False,
        )
    except Exception as error:
        raise Gen3SamplingError(f"Molexar generation failed: {error}") from error
    if len(fragment_rows) != n or len(converted) != n:
        raise Gen3SamplingError(
            f"Molexar returned {len(fragment_rows)} draws and {len(converted)} "
            f"conversions for a request of {n}"
        )
    records = []
    for fragment_selfies, pair in zip(fragment_rows, converted):
        converted_fragment, smiles = pair
        if converted_fragment != fragment_selfies:
            raise Gen3SamplingError("Molexar conversion changed raw row ordering")
        raw_smiles = "" if smiles is None else str(smiles)
        records.append(
            {
                "fragment_selfies": str(fragment_selfies),
                "raw_smiles": raw_smiles,
                "conversion_success": smiles is not None,
                "fragment_selfies_character_length": len(str(fragment_selfies)),
                "smiles_character_length": len(raw_smiles),
            }
        )
    return records














def _transport_smiles(raw_smiles: str) -> tuple[str, str]:              # When Fragment-SELFIES fails to decode
    if raw_smiles == "":                                                # you get an empty string.
        return EMPTY_TRANSPORT_TOKEN, "empty_smiles"                    # An empty line in a .smi file just... disappears
    if any(character.isspace() for character in raw_smiles):
        return WHITESPACE_TRANSPORT_TOKEN, "contains_whitespace"        # so instead just replace it with a deliberately invalid placeholder.
    return raw_smiles, ""                                               # The intake pipeline can then decide what to do with it












def generate_gen3_samples(
    model_dir: Path,
    target_pdb: Path,
    outdir: Path,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    n: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Write one immutable, authenticated Gen3 raw cohort."""

    started = time.perf_counter()
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen3SamplingError(f"Gen3 output directory already exists: {outdir}")
    specification = load_gen3_config(config_path)
    sampling = dict(specification["sampling"])
    resolved_n = int(sampling["raw_samples"] if n is None else n)
    resolved_seed = int(sampling["seed"] if seed is None else seed)
    target_record = verify_target(target_pdb, specification)
    engine, checkpoint = load_model_bundle(
        model_dir, source_root, device=device, specification=specification
    )
    pocket_graph, pocket_record = build_pocket_graph(
        engine, target_pdb, specification
    )
    records = sample_molecules(
        engine,
        pocket_graph,
        n=resolved_n,
        seed=resolved_seed,
        sampling=sampling,
    )

    outdir.mkdir(parents=True)
    raw_path = outdir / RAW_SAMPLES_NAME
    smi_path = outdir / MOLECULES_NAME
    raw_temporary = raw_path.with_name(f".{raw_path.name}.tmp")
    smi_lines: list[str] = []
    transport_counts: dict[str, int] = {}
    with raw_temporary.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "sample_index",
            "molecule_id",
            "fragment_selfies",
            "raw_smiles",
            "transport_smiles",
            "transport_encoding",
            "conversion_success",
            "fragment_selfies_character_length",
            "smiles_character_length",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, record in enumerate(records, 1):
            molecule_id = f"GEN3_S{resolved_seed}_{index:05d}"
            transport_smiles, transport_encoding = _transport_smiles(
                str(record["raw_smiles"])
            )
            if transport_encoding:
                transport_counts[transport_encoding] = (
                    transport_counts.get(transport_encoding, 0) + 1
                )
            writer.writerow(
                {
                    "sample_index": index,
                    "molecule_id": molecule_id,
                    **record,
                    "transport_smiles": transport_smiles,
                    "transport_encoding": transport_encoding,
                }
            )
            smi_lines.append(f"{transport_smiles} {molecule_id}\n")
    raw_temporary.replace(raw_path)
    temporary_smi = smi_path.with_name(f".{smi_path.name}.tmp")
    temporary_smi.write_text("".join(smi_lines), encoding="utf-8")
    temporary_smi.replace(smi_path)

    fragment_lengths = np.asarray(
        [row["fragment_selfies_character_length"] for row in records], dtype=float
    )
    smiles_lengths = np.asarray(
        [row["smiles_character_length"] for row in records], dtype=float
    )
    conversion_successes = sum(bool(row["conversion_success"]) for row in records)
    summary = {
        "schema_version": GEN3_SCHEMA_VERSION,
        "stage": "gen3_sampling",
        "model_name": MODEL_NAME,
        "interpretation": {
            "target_aware": True,
            "conditioning": "locked 7O7K protein-pocket coordinates only",
            "target_disjoint_training_claimed": False,
            "local_training_or_fine_tuning": False,
            "all_raw_draws_retained": True,
            "validity_filtering": False,
            "deduplication": False,
            "replacement_or_top_up": False,
            "active_ligand_or_docking_reward_used": False,
        },
        "parameters": {
            **sampling,
            "seed": resolved_seed,
            "raw_samples": resolved_n,
            "n_samples": resolved_n,
            "device": str(engine.device),
        },
        "target": specification["target"],
        "pocket_graph": pocket_record,
        "checkpoint": checkpoint,
        "provenance": {
            "configuration": runtime.file_record(Path(config_path)),
            "target_pdb": target_record,
        },
        "counts": {
            "raw_samples": len(records),
            "fragment_selfies_conversion_successes": conversion_successes,
            "fragment_selfies_conversion_failures": len(records) - conversion_successes,
            "empty_raw_smiles": sum(row["raw_smiles"] == "" for row in records),
            "transport_encodings": dict(sorted(transport_counts.items())),
        },
        "raw_fragment_selfies_character_length": _length_statistics(fragment_lengths),
        "converted_smiles_character_length": _length_statistics(smiles_lengths),
        "outputs": {
            "raw_samples_csv": runtime.file_record(raw_path),
            "molecules_smi": runtime.file_record(smi_path),
        },
        "software": checkpoint["runtime"],
        "hardware": runtime.hardware_record(),
        "timing": runtime.timing_record(started, attempted_tasks=resolved_n, workers=1),
    }
    runtime.write_json_atomic(outdir / SAMPLING_NAME, summary)
    return summary


def _length_statistics(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": int(np.min(values)),
        "maximum": int(np.max(values)),
    }




















def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample frozen Molexar using only the registered 7O7K pocket"
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("target_pdb", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        summary = generate_gen3_samples(
            args.model_dir,
            args.target_pdb,
            args.outdir,
            source_root=args.source_root,
            config_path=args.config,
            n=args.n,
            seed=args.seed,
            device=args.device,
        )
    except (
        Gen3CheckpointError,
        Gen3ConfigurationError,
        Gen3SamplingError,
    ) as error:
        parser.error(str(error))
    print(
        f"Gen3 sampling complete: {summary['counts']['raw_samples']:,} raw draws; "
        f"{summary['counts']['fragment_selfies_conversion_successes']:,} converted; "
        f"outputs: {Path(args.outdir).resolve()}"
    )


if __name__ == "__main__":
    main()
