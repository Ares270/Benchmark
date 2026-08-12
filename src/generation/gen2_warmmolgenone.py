"""Sample the frozen, protein-sequence-conditioned WarmMolGenOne checkpoint.

This module is the first implementation layer for Gen2.  It authenticates a
local copy of the official checkpoint, validates the locked DYRK1A target and
sampling configuration, and retains every raw decoded string.  It does not
train, fine-tune, filter, rank, deduplicate, replace, or top up molecules.

The Transformers dependency is imported lazily so the established docking and
analysis environment remains usable while the Gen2 runtime is being validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.harness import config, runtime


GEN2_SCHEMA_VERSION = 1
MODEL_NAME = "gen2_warmmolgenone_v1"
DEFAULT_CONFIG_PATH = config.REPO_ROOT / "configs/gen2_warmmolgenone.json"
RAW_SAMPLES_NAME = "raw_samples.csv"
MOLECULES_NAME = "molecules.smi"
SAMPLING_NAME = "sampling.json"
EMPTY_TRANSPORT_TOKEN = "__GEN2_EMPTY_SMILES__"
WHITESPACE_TRANSPORT_TOKEN = "__GEN2_WHITESPACE_SMILES__"


class Gen2ConfigurationError(ValueError):
    """Raised when the locked Gen2 scientific configuration is inconsistent."""


class Gen2CheckpointError(ValueError):
    """Raised when local model artifacts do not match the locked checkpoint."""


class Gen2SamplingError(ValueError):
    """Raised when a sampling request or runtime violates the raw-output contract."""


def load_gen2_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read and validate the versioned Gen2 scientific configuration."""

    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gen2ConfigurationError(f"Cannot read Gen2 config {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gen2ConfigurationError("Gen2 config must contain one JSON object")
    if value.get("schema_version") != GEN2_SCHEMA_VERSION:
        raise Gen2ConfigurationError("Unsupported Gen2 config schema version")
    if value.get("arm", {}).get("name") != MODEL_NAME:
        raise Gen2ConfigurationError("Gen2 config arm name does not match the implementation")

    target = value.get("target", {})
    sequence = str(target.get("sequence", ""))
    if len(sequence) != int(target.get("input_sequence_length", -1)):
        raise Gen2ConfigurationError("Gen2 target sequence length does not match the config")
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise Gen2ConfigurationError("Gen2 target must contain only 20 canonical amino acids")
    observed_hash = hashlib.sha256(f"{sequence}\n".encode("utf-8")).hexdigest()
    if observed_hash != target.get("sequence_utf8_lf_sha256"):
        raise Gen2ConfigurationError("Gen2 target sequence SHA-256 does not match the config")

    sampling = value.get("sampling", {})
    integer_fields = ("seed", "raw_samples", "docking_subsample", "batch_size")
    if any(int(sampling.get(field, -1)) < 1 for field in integer_fields):
        raise Gen2ConfigurationError("Gen2 seed, counts, and batch size must be positive")
    if int(sampling["docking_subsample"]) > int(sampling["raw_samples"]):
        raise Gen2ConfigurationError("Gen2 docking subsample cannot exceed raw samples")
    if sampling.get("validity_filtering") or sampling.get("deduplication"):
        raise Gen2ConfigurationError("Gen2 generator filtering and deduplication must remain off")
    if sampling.get("replacement_or_top_up"):
        raise Gen2ConfigurationError("Gen2 replacement/top-up must remain off")
    return value


def _verify_artifacts(
    directory: Path,
    artifacts: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for filename, expected in artifacts.items():
        path = directory / filename
        if not path.is_file():
            raise Gen2CheckpointError(f"{label} lacks {filename}: {path}")
        observed_bytes = path.stat().st_size
        if observed_bytes != int(expected["bytes"]):
            raise Gen2CheckpointError(
                f"{label} {filename} byte count mismatch: expected "
                f"{expected['bytes']}, observed {observed_bytes}"
            )
        observed_hash = runtime.sha256_file(path)
        if observed_hash != expected["sha256"]:
            raise Gen2CheckpointError(
                f"{label} {filename} SHA-256 mismatch: expected "
                f"{expected['sha256']}, observed {observed_hash}"
            )
        records[filename] = runtime.file_record(path)
    return records


def verify_checkpoint(model_dir: Path, specification: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the official weight file before importing Transformers."""

    model_dir = Path(model_dir)
    model = specification["model"]
    weights = model_dir / str(model["weights_file"])
    if not weights.is_file():
        raise Gen2CheckpointError(f"WarmMolGenOne weights do not exist: {weights}")
    observed_bytes = weights.stat().st_size
    if observed_bytes != int(model["weights_bytes"]):
        raise Gen2CheckpointError(
            "WarmMolGenOne checkpoint byte count mismatch: "
            f"expected {model['weights_bytes']}, observed {observed_bytes}"
        )
    observed_hash = runtime.sha256_file(weights)
    if observed_hash != model["weights_sha256"]:
        raise Gen2CheckpointError(
            "WarmMolGenOne checkpoint SHA-256 mismatch: "
            f"expected {model['weights_sha256']}, observed {observed_hash}"
        )
    artifacts = _verify_artifacts(
        model_dir, model["artifacts"], label="WarmMolGenOne model directory"
    )
    return {
        "declared_model": model,
        "weights": runtime.file_record(weights),
        "artifacts": artifacts,
        "local_model_dir": str(model_dir.resolve()),
    }


def _transformers_classes() -> tuple[Any, Any]:
    try:
        from transformers import EncoderDecoderModel, RobertaTokenizer
    except ImportError as error:
        raise Gen2CheckpointError(
            "Gen2 requires the pinned Transformers runtime; install it only after "
            "reviewing the Gen2 setup instructions"
        ) from error
    return EncoderDecoderModel, RobertaTokenizer


def load_model_bundle(
    model_dir: Path,
    decoder_tokenizer_dir: Path,
    *,
    device: str,
    specification: dict[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load authenticated local model and tokenizers without network fallback."""

    provenance = verify_checkpoint(model_dir, specification)
    decoder_tokenizer_dir = Path(decoder_tokenizer_dir)
    decoder_artifacts = _verify_artifacts(
        decoder_tokenizer_dir,
        specification["tokenizers"]["smiles"]["artifacts"],
        label="SMILES tokenizer directory",
    )

    EncoderDecoderModel, RobertaTokenizer = _transformers_classes()
    try:
        protein_tokenizer = RobertaTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )
        smiles_tokenizer = RobertaTokenizer.from_pretrained(
            str(decoder_tokenizer_dir), local_files_only=True
        )
        model = EncoderDecoderModel.from_pretrained(
            str(model_dir), local_files_only=True, weights_only=True
        )
        resolved_device = torch.device(device)
        model = model.to(resolved_device)
    except Exception as error:
        raise Gen2CheckpointError(f"Cannot load WarmMolGenOne locally: {error}") from error

    encoder = model.config.encoder
    decoder = model.config.decoder
    observed = {
        "encoder_layers": int(encoder.num_hidden_layers),
        "decoder_layers": int(decoder.num_hidden_layers),
        "encoder_hidden_size": int(encoder.hidden_size),
        "decoder_hidden_size": int(decoder.hidden_size),
        "encoder_vocab_size": int(encoder.vocab_size),
        "decoder_vocab_size": int(decoder.vocab_size),
    }
    expected = {
        "encoder_layers": 12,
        "decoder_layers": 6,
        "encoder_hidden_size": 768,
        "decoder_hidden_size": 768,
        "encoder_vocab_size": 10261,
        "decoder_vocab_size": 52000,
    }
    if observed != expected:
        raise Gen2CheckpointError(
            f"WarmMolGenOne architecture mismatch: expected {expected}, observed {observed}"
        )
    model.eval()
    provenance["observed_architecture"] = observed
    provenance["decoder_tokenizer_dir"] = str(decoder_tokenizer_dir.resolve())
    provenance["decoder_tokenizer_artifacts"] = decoder_artifacts
    return model, protein_tokenizer, smiles_tokenizer, provenance


def _seed_everything(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise Gen2SamplingError("Cannot determine the Gen2 model device") from error


def sample_smiles(
    model: Any,
    protein_tokenizer: Any,
    smiles_tokenizer: Any,
    target_sequence: str,
    *,
    n: int,
    seed: int,
    sampling: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exactly ``n`` unfiltered stochastic decoder outputs."""

    if n < 1:
        raise Gen2SamplingError("n must be at least 1")
    if seed < 0:
        raise Gen2SamplingError("seed must be non-negative")
    batch_size = int(sampling["batch_size"])
    max_length = int(sampling["max_length_decoder_tokens"])
    if batch_size < 1 or max_length < 2:
        raise Gen2SamplingError("batch size and decoder maximum must be positive")

    device = _model_device(model)
    _seed_everything(seed, device)
    try:
        encoded = protein_tokenizer(target_sequence, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
    except Exception as error:
        raise Gen2SamplingError(f"Cannot tokenize the Gen2 target sequence: {error}") from error

    records: list[dict[str, Any]] = []
    eos_id = int(smiles_tokenizer.eos_token_id)
    bos_id = int(smiles_tokenizer.bos_token_id)
    model.eval()
    with torch.inference_mode():
        while len(records) < n:
            current_batch = min(batch_size, n - len(records))
            try:
                outputs = model.generate(
                    **encoded,
                    decoder_start_token_id=bos_id,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    max_length=max_length,
                    num_return_sequences=current_batch,
                    do_sample=True,
                    temperature=float(sampling["temperature"]),
                    top_k=int(sampling["top_k"]),
                    top_p=float(sampling["top_p"]),
                    num_beams=int(sampling["num_beams"]),
                )
                decoded = smiles_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
            except Exception as error:
                raise Gen2SamplingError(f"WarmMolGenOne generation failed: {error}") from error
            if len(decoded) != current_batch:
                raise Gen2SamplingError(
                    "WarmMolGenOne returned a different number of rows than requested"
                )
            for token_row, raw_smiles in zip(outputs.detach().cpu().tolist(), decoded):
                terminated = eos_id in token_row[1:]
                records.append(
                    {
                        "raw_smiles": str(raw_smiles),
                        "token_count": len(token_row),
                        "character_length": len(str(raw_smiles)),
                        "terminated_by_eos": terminated,
                        "hit_max_length": not terminated and len(token_row) >= max_length,
                    }
                )
    return records


def _transport_smiles(raw_smiles: str) -> tuple[str, str]:
    if raw_smiles == "":
        return EMPTY_TRANSPORT_TOKEN, "empty_smiles"
    if any(character.isspace() for character in raw_smiles):
        return WHITESPACE_TRANSPORT_TOKEN, "contains_whitespace"
    return raw_smiles, ""


def generate_gen2_samples(
    model_dir: Path,
    decoder_tokenizer_dir: Path,
    outdir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    n: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Write one immutable Gen2 raw cohort and its complete provenance."""

    started = time.perf_counter()
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen2SamplingError(f"Gen2 output directory already exists: {outdir}")
    config_path = Path(config_path)
    specification = load_gen2_config(config_path)
    sampling = dict(specification["sampling"])
    resolved_n = int(sampling["raw_samples"] if n is None else n)
    resolved_seed = int(sampling["seed"] if seed is None else seed)
    model, protein_tokenizer, smiles_tokenizer, checkpoint = load_model_bundle(
        model_dir,
        decoder_tokenizer_dir,
        device=device,
        specification=specification,
    )
    records = sample_smiles(
        model,
        protein_tokenizer,
        smiles_tokenizer,
        specification["target"]["sequence"],
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
            "sample_index", "molecule_id", "raw_smiles", "transport_smiles",
            "transport_encoding", "token_count", "character_length",
            "terminated_by_eos", "hit_max_length",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, record in enumerate(records, 1):
            molecule_id = f"GEN2_S{resolved_seed}_{index:05d}"
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
                    "raw_smiles": record["raw_smiles"],
                    "transport_smiles": transport_smiles,
                    "transport_encoding": transport_encoding,
                    "token_count": record["token_count"],
                    "character_length": record["character_length"],
                    "terminated_by_eos": record["terminated_by_eos"],
                    "hit_max_length": record["hit_max_length"],
                }
            )
            smi_lines.append(f"{transport_smiles} {molecule_id}\n")
    raw_temporary.replace(raw_path)
    smi_temporary = smi_path.with_name(f".{smi_path.name}.tmp")
    smi_temporary.write_text("".join(smi_lines), encoding="utf-8")
    smi_temporary.replace(smi_path)

    lengths = np.asarray([row["character_length"] for row in records], dtype=float)
    summary = {
        "schema_version": GEN2_SCHEMA_VERSION,
        "stage": "gen2_sampling",
        "model_name": MODEL_NAME,
        "interpretation": {
            "target_aware": True,
            "conditioning": "locked DYRK1A protein sequence only",
            "target_disjoint_training_claimed": False,
            "local_training_or_fine_tuning": False,
            "all_raw_draws_retained": True,
            "validity_filtering": False,
            "deduplication": False,
            "replacement_or_top_up": False,
        },
        "parameters": {
            **sampling,
            "seed": resolved_seed,
            "raw_samples": resolved_n,
            "n_samples": resolved_n,
            "device": str(_model_device(model)),
        },
        "target": specification["target"],
        "checkpoint": checkpoint,
        "provenance": {
            "configuration": runtime.file_record(config_path),
            "target_structure": runtime.file_record(
                config.REPO_ROOT / specification["target"]["source_file"]
            ),
        },
        "counts": {
            "raw_samples": len(records),
            "terminated_by_eos": sum(bool(row["terminated_by_eos"]) for row in records),
            "hit_max_length": sum(bool(row["hit_max_length"]) for row in records),
            "empty_raw_smiles": sum(row["raw_smiles"] == "" for row in records),
            "transport_encodings": dict(sorted(transport_counts.items())),
        },
        "raw_character_length": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "minimum": int(np.min(lengths)),
            "maximum": int(np.max(lengths)),
        },
        "outputs": {
            "raw_samples_csv": runtime.file_record(raw_path),
            "molecules_smi": runtime.file_record(smi_path),
        },
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "huggingface_hub": package_version("huggingface-hub"),
            "safetensors": package_version("safetensors"),
        },
        "hardware": runtime.hardware_record(),
        "timing": runtime.timing_record(started, attempted_tasks=resolved_n, workers=1),
    }
    runtime.write_json_atomic(outdir / SAMPLING_NAME, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the frozen protein-conditioned WarmMolGenOne Gen2 model"
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("decoder_tokenizer_dir", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        summary = generate_gen2_samples(
            args.model_dir,
            args.decoder_tokenizer_dir,
            args.outdir,
            config_path=args.config,
            n=args.n,
            seed=args.seed,
            device=args.device,
        )
    except (Gen2ConfigurationError, Gen2CheckpointError, Gen2SamplingError) as error:
        parser.error(str(error))
    print(
        f"Gen2 sampling complete: {summary['counts']['raw_samples']:,} raw draws; "
        f"outputs: {Path(args.outdir).resolve()}"
    )


if __name__ == "__main__":
    main()
