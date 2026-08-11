"""Sample the frozen, target-unaware GuacaMol SMILES-LSTM checkpoint.

This is a compatibility implementation of the historical GuacaMol baseline at
commit ``ae43219c89d5db134028336243f508606d81995e``. It does not train or
fine-tune a model. The official checkpoint is authenticated by byte count and
SHA-256, loaded with ``weights_only=True`` and ``strict=True``, and sampled
with the original vocabulary and ActionSampler semantics.

Every raw draw is retained. There is no validity filtering, deduplication,
replacement, ranking, or top-up in this module.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Categorical

from src.harness import runtime


GEN1_SCHEMA_VERSION = 1
MODEL_NAME = "gen1_guacamol_smiles_lstm_v1"
OFFICIAL_CHECKPOINT_FILENAME = "model_final_0.473.pt"
OFFICIAL_CHECKPOINT_BYTES = 101_148_783
OFFICIAL_CHECKPOINT_SHA256 = (
    "da1be33b519868ba3a0befabe5b018d247bfd65ccaae7d5fdac4c804f23580f3"
)
UPSTREAM_COMMIT = "ae43219c89d5db134028336243f508606d81995e"
RAW_SAMPLES_NAME = "raw_samples.csv"
MOLECULES_NAME = "molecules.smi"
SAMPLING_NAME = "sampling.json"

DECLARED_SEED = 20260801
DECLARED_SAMPLE_SIZE = 10_000
DECLARED_DOCKING_SUBSAMPLE_SIZE = 1_000

ARCHITECTURE = {
    "input_size": 47,
    "hidden_size": 1024,
    "output_size": 47,
    "num_layers": 3,
    "rnn_dropout": 0.2,
    "cell": "LSTM",
    "tokenization": "GuacaMol SmilesCharDictionary",
}

TRAINING = {
    "training_universe": "GuacaMol v1 training SMILES (ChEMBL 24)",
    "target_information_used": False,
    "local_training_or_fine_tuning": False,
    "provenance_limit": (
        "The distributed checkpoint and repository identify the training corpus "
        "and architecture; no additional training history is inferred."
    ),
}

SAMPLING = {
    "distribution": "full-vocabulary categorical softmax",
    "temperature": 1.0,
    "top_k": None,
    "top_p": None,
    "beam_search": False,
    "max_length_characters": 100,
    "batch_size": 64,
    "invalid_replacement": False,
    "duplicate_replacement": False,
    "post_generation_filtering": False,
}

OFFICIAL_SOURCE = {
    "description": (
        "Official GuacaMol SMILES-LSTM distribution-learning baseline checkpoint "
        "model_final_0.473.pt; trained on the GuacaMol v1 training set derived "
        "from ChEMBL 24; target-unaware with respect to DYRK1A"
    ),
    "repository": "https://github.com/BenevolentAI/guacamol_baselines",
    "repository_commit": UPSTREAM_COMMIT,
    "checkpoint_url": (
        "https://github.com/BenevolentAI/guacamol_baselines/blob/"
        f"{UPSTREAM_COMMIT}/smiles_lstm_hc/pretrained_model/"
        f"{OFFICIAL_CHECKPOINT_FILENAME}"
    ),
    "model_config_url": (
        "https://github.com/BenevolentAI/guacamol_baselines/blob/"
        f"{UPSTREAM_COMMIT}/smiles_lstm_hc/pretrained_model/"
        "model_final_0.473.json"
    ),
    "guacamol_publication": "https://doi.org/10.1021/acs.jcim.8b00839",
    "license": "MIT",
}

EMPTY_TRANSPORT_TOKEN = "__GEN1_EMPTY_SMILES__"
WHITESPACE_TRANSPORT_TOKEN = "__GEN1_WHITESPACE_SMILES__"


class Gen1CheckpointError(ValueError):
    """Raised when the official Gen1 checkpoint cannot be authenticated."""


class Gen1SamplingError(ValueError):
    """Raised when a sampling request violates the raw-output contract."""


@dataclass(frozen=True)
class GuacaMolVocabulary:
    """Exact historical ``SmilesCharDictionary`` used by the checkpoint."""

    pad: str = " "
    begin: str = "Q"
    end: str = "\n"

    def __post_init__(self) -> None:
        char_idx = {
            " ": 0, "Q": 1, "\n": 2, "#": 20, "%": 22, "(": 25,
            ")": 24, "+": 26, "-": 27, ".": 30, "0": 32, "1": 31,
            "2": 34, "3": 33, "4": 36, "5": 35, "6": 38, "7": 37,
            "8": 40, "9": 39, "=": 41, "A": 7, "B": 11, "C": 19,
            "F": 4, "H": 6, "I": 5, "N": 10, "O": 9, "P": 12,
            "S": 13, "X": 15, "Y": 14, "Z": 3, "[": 16, "]": 18,
            "b": 21, "c": 8, "n": 17, "o": 29, "p": 23, "s": 28,
            "@": 42, "R": 43, "/": 44, "\\": 45, "E": 46,
        }
        object.__setattr__(self, "char_idx", char_idx)
        object.__setattr__(
            self, "idx_char", {value: key for key, value in char_idx.items()}
        )
        decode_dict = {
            "Y": "Br", "X": "Cl", "A": "Si", "Z": "Se", "R": "@@", "E": "se"
        }
        object.__setattr__(self, "decode_dict", decode_dict)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[self.begin]

    @property
    def end_idx(self) -> int:
        return self.char_idx[self.end]

    def decode_tokens(self, token_ids: list[int]) -> str:
        encoded = "".join(self.idx_char[token_id] for token_id in token_ids)
        for token, symbol in self.decode_dict.items():
            encoded = encoded.replace(token, symbol)
        return encoded


class GuacaMolSmilesLSTM(nn.Module):
    """Tensor-name-compatible implementation of the official ``SmilesRnn``."""

    def __init__(
        self,
        input_size: int = int(ARCHITECTURE["input_size"]),
        hidden_size: int = int(ARCHITECTURE["hidden_size"]),
        output_size: int = int(ARCHITECTURE["output_size"]),
        num_layers: int = int(ARCHITECTURE["num_layers"]),
        rnn_dropout: float = float(ARCHITECTURE["rnn_dropout"]),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.encoder(token_ids)
        output, hidden = self.rnn(embedded, hidden)
        return self.decoder(output), hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (self.num_layers, batch_size, self.hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


def _resolve_checkpoint(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / OFFICIAL_CHECKPOINT_FILENAME
    if not checkpoint.is_file():
        raise Gen1CheckpointError(f"Gen1 checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_checkpoint(
    checkpoint: Path,
    *,
    device: str = "cpu",
) -> tuple[GuacaMolSmilesLSTM, GuacaMolVocabulary, dict]:
    """Authenticate and strictly load the official GuacaMol checkpoint."""

    checkpoint = _resolve_checkpoint(checkpoint)
    actual_bytes = checkpoint.stat().st_size
    if actual_bytes != OFFICIAL_CHECKPOINT_BYTES:
        raise Gen1CheckpointError(
            f"Checkpoint byte count mismatch for {checkpoint}: expected "
            f"{OFFICIAL_CHECKPOINT_BYTES}, got {actual_bytes}"
        )
    actual_hash = runtime.sha256_file(checkpoint)
    if actual_hash != OFFICIAL_CHECKPOINT_SHA256:
        raise Gen1CheckpointError(
            f"Checkpoint SHA-256 mismatch for {checkpoint}: expected "
            f"{OFFICIAL_CHECKPOINT_SHA256}, got {actual_hash}"
        )

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Gen1CheckpointError(
            f"Cannot safely load Gen1 weights {checkpoint}: {error}"
        ) from error
    if not isinstance(state, dict) or not state:
        raise Gen1CheckpointError("GuacaMol checkpoint must be a non-empty state dict")

    model = GuacaMolSmilesLSTM()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Gen1CheckpointError(
            f"Checkpoint tensors do not match the official architecture: {error}"
        ) from error
    try:
        resolved_device = torch.device(device)
        model = model.to(resolved_device)
    except (RuntimeError, ValueError) as error:
        raise Gen1CheckpointError(
            f"Cannot place Gen1 model on {device!r}: {error}"
        ) from error
    model.eval()
    provenance = {
        "declared_source": OFFICIAL_SOURCE,
        "weights": runtime.file_record(checkpoint),
        "expected_bytes": OFFICIAL_CHECKPOINT_BYTES,
        "expected_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "architecture": ARCHITECTURE,
        "training": TRAINING,
    }
    return model, GuacaMolVocabulary(), provenance


def _seed_everything(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def sample_smiles(
    model: GuacaMolSmilesLSTM,
    vocabulary: GuacaMolVocabulary,
    *,
    n: int,
    seed: int,
    max_length: int = int(SAMPLING["max_length_characters"]),
    batch_size: int = int(SAMPLING["batch_size"]),
) -> list[dict]:
    """Return exactly ``n`` draws using the historical ActionSampler algorithm."""

    if n < 1:
        raise Gen1SamplingError("n must be at least 1")
    if seed < 0:
        raise Gen1SamplingError("seed must be non-negative")
    if max_length < 1 or batch_size < 1:
        raise Gen1SamplingError("max_length and batch_size must be at least 1")

    device = next(model.parameters()).device
    _seed_everything(seed, device)
    records: list[dict] = []
    model.eval()

    with torch.inference_mode():
        while len(records) < n:
            current_batch = min(batch_size, n - len(records))
            hidden = model.init_hidden(current_batch, device)
            token = torch.full(
                (current_batch, 1),
                vocabulary.begin_idx,
                dtype=torch.long,
                device=device,
            )
            actions = torch.zeros(
                (current_batch, max_length), dtype=torch.long, device=device
            )
            for character_index in range(max_length):
                output, hidden = model(token, hidden)
                probabilities = functional.softmax(output, dim=2)
                action = Categorical(probs=probabilities).sample()
                actions[:, character_index] = action.squeeze(1)
                token = action

            for row in actions.cpu().tolist():
                terminated = vocabulary.end_idx in row
                token_ids = row[: row.index(vocabulary.end_idx)] if terminated else row
                smiles = vocabulary.decode_tokens(token_ids)
                records.append(
                    {
                        "raw_smiles": smiles,
                        "token_count": len(token_ids),
                        "character_length": len(smiles),
                        "terminated_by_eos": terminated,
                        "hit_max_length": not terminated,
                    }
                )
    return records


def _transport_smiles(raw_smiles: str) -> tuple[str, str]:
    if raw_smiles == "":
        return EMPTY_TRANSPORT_TOKEN, "empty_smiles"
    if any(character.isspace() for character in raw_smiles):
        return WHITESPACE_TRANSPORT_TOKEN, "contains_whitespace"
    return raw_smiles, ""


def generate_gen1_samples(
    checkpoint: Path,
    outdir: Path,
    *,
    n: int = DECLARED_SAMPLE_SIZE,
    seed: int = DECLARED_SEED,
    device: str = "cpu",
) -> dict:
    """Load the checkpoint and write one immutable, unfiltered raw cohort."""

    started = time.perf_counter()
    outdir = Path(outdir)
    if outdir.exists():
        raise Gen1SamplingError(f"Gen1 output directory already exists: {outdir}")

    model, vocabulary, checkpoint_record = load_checkpoint(checkpoint, device=device)
    records = sample_smiles(model, vocabulary, n=n, seed=seed)
    outdir.mkdir(parents=True)
    raw_path = outdir / RAW_SAMPLES_NAME
    smi_path = outdir / MOLECULES_NAME

    raw_temporary = raw_path.with_name(f".{raw_path.name}.tmp")
    smi_lines = []
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
            molecule_id = f"GEN1_S{seed}_{index:05d}"
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

    lengths = np.asarray(
        [record["character_length"] for record in records], dtype=float
    )
    summary = {
        "schema_version": GEN1_SCHEMA_VERSION,
        "stage": "gen1_sampling",
        "model_name": MODEL_NAME,
        "interpretation": {
            "target_aware": False,
            "target_information_used_by_model": False,
            "target_disjoint_training_claimed": False,
            "all_raw_draws_retained": True,
            "validity_filtering": False,
            "deduplication": False,
            "replacement_or_top_up": False,
            "transport_note": (
                "Empty or whitespace-containing raw strings remain authoritative in "
                "raw_samples.csv and use explicit invalid sentinel strings only in "
                "the two-column molecules.smi transport file."
            ),
        },
        "parameters": {
            "seed": seed,
            "n_samples": n,
            "device": str(next(model.parameters()).device),
            **SAMPLING,
        },
        "counts": {
            "raw_samples": len(records),
            "terminated_by_eos": sum(
                record["terminated_by_eos"] for record in records
            ),
            "hit_max_length": sum(record["hit_max_length"] for record in records),
            "empty_raw_smiles": sum(
                record["raw_smiles"] == "" for record in records
            ),
            "transport_encodings": dict(sorted(transport_counts.items())),
        },
        "raw_character_length": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "minimum": int(np.min(lengths)),
            "maximum": int(np.max(lengths)),
        },
        "checkpoint": checkpoint_record,
        "outputs": {
            "raw_samples_csv": runtime.file_record(raw_path),
            "molecules_smi": runtime.file_record(smi_path),
        },
        "software": {"torch": torch.__version__, "numpy": np.__version__},
        "hardware": runtime.hardware_record(),
        "timing": runtime.timing_record(started, attempted_tasks=n, workers=1),
    }
    runtime.write_json_atomic(outdir / SAMPLING_NAME, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the official target-unaware GuacaMol SMILES-LSTM"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--n", type=int, default=DECLARED_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DECLARED_SEED)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        summary = generate_gen1_samples(
            args.checkpoint,
            args.outdir,
            n=args.n,
            seed=args.seed,
            device=args.device,
        )
    except (Gen1CheckpointError, Gen1SamplingError) as error:
        parser.error(str(error))
    print(
        f"Gen1 sampling complete: {summary['counts']['raw_samples']:,} raw draws; "
        f"outputs: {Path(args.outdir).resolve()}"
    )


if __name__ == "__main__":
    main()
