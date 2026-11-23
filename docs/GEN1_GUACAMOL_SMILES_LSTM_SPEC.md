# Gen1 target-unaware GuacaMol SMILES-LSTM

**Decision date:** 2026-08-10  
**Status:** official checkpoint authenticated; compatibility loader and pipeline
validation implemented  

## 1. Scientific role

Gen1 is the target-unaware model arm. It measures what a generic pretrained
molecular language model contributes before any target-conditioned model is
introduced.

The sampler receives no DYRK1A actives, inactive set, receptor, docking score,
pharmacophore, similarity reward, or target-conditioned fine-tuning. Its
training corpus is ChEMBL-derived, so target-unaware must not be misreported as
target-disjoint. Exact known-active parent rediscoveries are computed after
generation with RDKit InChIKeys and are reported, never filtered or replaced.

## 2. Registered one-run design

There is one model run:

- root seed: `20260801`;
- raw model draws: exactly `10,000`;
- generation-quality analysis: all `10,000`;
- docking branch: a uniform-without-replacement raw sample of exactly `1,000`;
- raw subsampling occurs before intake, validity checking, parent
  deduplication, or the frozen pre-dock gate;
- no invalid, duplicate, gate-excluded, preparation-failed, or docking-failed
  molecule is replaced or topped up.

```text
official checkpoint -> 10,000 raw SMILES -> full-cohort intake and analysis
                                         -> uniform raw 1,000
                                            -> intake
                                            -> frozen pre-dock gate
                                            -> ligand preparation
                                            -> Smina docking
                                            -> candidate-only analysis
                                            -> combined Gen1 report
```

Small runs are permitted only with `--validation`. They use role `pilot`, carry
`registered_campaign: false`, and do not constitute the model-arm result.

## 3. Authenticated checkpoint

Local path:

```text
Models & Miscellaneous/model_final_0.473.pt
```

The entire local directory is Git-ignored. The checkpoint identity is frozen:

| Field | Value |
| --- | --- |
| Filename | `model_final_0.473.pt` |
| Bytes | `101148783` |
| SHA-256 | `da1be33b519868ba3a0befabe5b018d247bfd65ccaae7d5fdac4c804f23580f3` |
| Upstream | `BenevolentAI/guacamol_baselines` |
| Pinned commit | `ae43219c89d5db134028336243f508606d81995e` |
| License | MIT |

Primary sources:

- [official checkpoint and model config](https://github.com/BenevolentAI/guacamol_baselines/tree/ae43219c89d5db134028336243f508606d81995e/smiles_lstm_hc/pretrained_model)
- [official model implementation](https://github.com/BenevolentAI/guacamol_baselines/blob/ae43219c89d5db134028336243f508606d81995e/smiles_lstm_hc/rnn_model.py)
- [official sampler implementation](https://github.com/BenevolentAI/guacamol_baselines/blob/ae43219c89d5db134028336243f508606d81995e/smiles_lstm_hc/action_sampler.py)
- [GuacaMol publication](https://doi.org/10.1021/acs.jcim.8b00839)

The loader rejects a different byte count or hash before deserialization. It
uses `torch.load(..., weights_only=True)` and `load_state_dict(...,
strict=True)`. A filename alone is not treated as authentication.

## 4. Exact compatibility contract

| Item | Value |
| --- | --- |
| Family | character-level autoregressive SMILES LSTM |
| Input / output vocabulary | exact historical 47-token `SmilesCharDictionary` |
| Embedding width | 1,024 |
| Recurrent layers | 3 |
| Hidden width | 1,024 |
| Inter-layer dropout | 0.2 (inactive in evaluation mode) |
| Training universe | GuacaMol v1 training SMILES, derived from ChEMBL 24 |
| Local training / fine-tuning | none |
| Sampling distribution | full-vocabulary categorical softmax |
| Temperature | 1.0 |
| Top-k / top-p / beam search | none |
| Sample batch | 64 |
| Decoding steps | 100 |
| Stop rule | returned SMILES ends at its first newline/EOS token |

The compatibility sampler intentionally performs all 100 recurrent steps for
every row, matching the historical `ActionSampler`, even when an EOS token was
already drawn. This affects the random-number stream and therefore
reproducibility. The returned SMILES is still truncated at the first EOS.

No unrecorded epochs, optimizer state, validation policy, or target-exclusion
claim is inferred from the weights. The pipeline records only provenance that
the distributed checkpoint and pinned upstream repository establish.

## 5. Raw-output and analysis contract

`raw_samples.csv` is authoritative and contains every decoder result, including
empty, invalid, and duplicate strings. `molecules.smi` is the two-column
pipeline transport file. Empty or whitespace-containing raw strings use an
explicit invalid sentinel in that transport file while their exact decoder
text remains in the CSV. Rows are never resampled.

The full-cohort report includes:

- EOS, maximum-length, empty-output, and length statistics;
- RDKit validity, unique valid parents, accepted fraction, and rejection
  reasons;
- multifragment and parent-extraction counts;
- exact known-active parent rediscoveries;
- Bemis-Murcko scaffold count and acyclic-parent count;
- MW, cLogP, TPSA, HBD, HBA, rotatable bonds, rings, fraction Csp3, formal
  charge, QED, and SA distributions;
- parent and scaffold novelty only when an authenticated GuacaMol v1 training
  reference index is supplied.

The screening report includes submitted, accepted at intake, passed gate,
prepared, successfully scored, survival rates, docking-score distribution,
ranking, and score/property correlations. The cohort is unlabelled, so AUC,
BEDROC, and enrichment are not applicable. Docking is a prioritization signal,
not proof of biochemical activity.

## 6. Commands

Small end-to-end validation:

```bash
python -m src.generation.run_gen1_pipeline "Models & Miscellaneous/model_final_0.473.pt" results/gen1_guacamol_validation_20260801 --validation --raw-n 100 --dock-n 10 --seed 20260801 --device cpu --workers 4
```

Registered Gen1 campaign:

```bash
python -m src.generation.run_gen1_pipeline "Models & Miscellaneous/model_final_0.473.pt" results/gen1_guacamol_smiles_lstm_v1_20260801 --seed 20260801 --device cpu --workers 4
```

Optional one-time novelty index from the authenticated, one-SMILES-per-line
GuacaMol v1 training file (`.gz` is accepted):

```bash
python -m src.generation.gen1_analysis build-reference-index PATH_TO_GUACAMOL_V1_TRAIN.smiles data/cache/guacamol_v1_train_reference --source-description "Official GuacaMol v1 training SMILES; exact URL and SHA-256 recorded separately"
```

Then append `--reference-index data/cache/guacamol_v1_train_reference` to the
campaign command. Without the index, novelty is marked unavailable and all
other analysis still runs. Output directories are immutable; a stage error is
fatal, and the runner does not repair, skip, replace, or top up molecules.
