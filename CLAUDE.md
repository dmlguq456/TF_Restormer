# CLAUDE.md — TF_Restormer

## Project Summary

Time-frequency domain speech enhancement system. Restores degraded speech (noisy, reverberant, clipped, bandwidth-limited) to high-quality wideband audio up to 48kHz. Uses hierarchical Transformer/Mamba architecture with Linformer attention, two-stage training (pretrain → adversarial), and multi-resolution processing.

## Documentation Coverage

| Document | Covers |
|---|---|
| [01_architecture_overview.md](.claude_reports/docs_code/01_architecture_overview.md) | Entry flow, module dependencies, design decisions |
| [02_model_network.md](.claude_reports/docs_code/02_model_network.md) | model.py, modules/module.py, modules/network.py, modules/msstftd.py |
| [03_engine.md](.claude_reports/docs_code/03_engine.md) | engine.py (Engine, EngineEval, EngineInfer, EngineInferFolder) |
| [04_dataset.md](.claude_reports/docs_code/04_dataset.md) | dataset.py (MyDataset, MyDatasetTest, get_dataloaders) |
| [05_loss.md](.claude_reports/docs_code/05_loss.md) | loss.py (all loss functions) |
| [06_utils.md](.claude_reports/docs_code/06_utils.md) | All utils/*.py files |
| [07_config_reference.md](.claude_reports/docs_code/07_config_reference.md) | baseline.yaml structure and registered datasets |

### File Coverage

| Source File | Doc |
|---|---|
| `run.py` | 01 |
| `models/TF_Restormer/main.py` | 01, 03 |
| `models/TF_Restormer/model.py` | 02 |
| `models/TF_Restormer/modules/module.py` | 02 |
| `models/TF_Restormer/modules/network.py` | 02 |
| `models/TF_Restormer/modules/msstftd.py` | 02 |
| `models/TF_Restormer/engine.py` | 03 |
| `models/TF_Restormer/dataset.py` | 04 |
| `models/TF_Restormer/loss.py` | 05 |
| `utils/util_stft.py` | 06 |
| `utils/util_dataset.py` | 06 |
| `utils/util_engine.py` | 06 |
| `utils/util_metric.py` | 06 |
| `utils/util_wvmos.py` | 06 |
| `utils/util_dnsmos.py` | 06 |
| `utils/util_nisqa.py` | 06 |
| `utils/util_composite.py` | 06 |
| `utils/util_mcd.py` | 06 |
| `utils/util_pesq.py` | 06 |
| `utils/util_writer.py` | 06 |
| `utils/util_system.py` | 06 |
| `utils/util_speechbertscore.py` | 06 |
| `utils/util_speechbleu.py` | 06 |
| `utils/util_speechtokendistance.py` | 06 |
| `utils/util_sBERTscore.py` | 06 |
| `utils/pos_embed.py` | 06 |
| `utils/decorators.py` | 06 |
| `utils/ASR_whisper.py` | 06 |
| `utils/ASR_w2v.py` | 06 |
| `utils/NISQA_models/NISQA_lib.py` | 06 |
| `utils/test_nisqa.py` | 06 |

## Project Structure

```
TF_Restormer_release/
├── run.py                              # CLI entry point
├── CLAUDE.md
├── models/TF_Restormer/
│   ├── main.py                         # Orchestrator
│   ├── model.py                        # Model_Enhance network
│   ├── engine.py                       # Train/Eval/Infer engines
│   ├── dataset.py                      # Data loading & synthesis
│   ├── loss.py                         # Loss functions
│   ├── modules/
│   │   ├── module.py                   # Encoder, Decoder, Upsampler
│   │   ├── network.py                  # Attention, Transformer, Mamba
│   │   └── msstftd.py                  # Multi-scale STFT discriminator
│   └── configs/
│       ├── baseline.yaml               # Main config
│       └── .legacy/                    # Archived configs (gitignored)
├── utils/                              # Shared utilities
│   ├── util_stft.py                    # STFT/iSTFT
│   ├── util_metric.py                  # PESQ, STOI, SDR, LSD, MCD
│   ├── util_engine.py                  # Checkpointing, schedulers
│   ├── util_dataset.py                 # SCP parsing, audio processing
│   ├── util_wvmos.py, util_dnsmos.py   # MOS metrics
│   ├── util_nisqa.py, util_composite.py
│   ├── util_writer.py                  # TensorBoard logging
│   ├── util_system.py                  # YAML, WandB
│   ├── ASR_whisper.py, ASR_w2v.py      # ASR utilities
│   └── NISQA_models/                   # NISQA checkpoints
├── data/
│   └── create_scp/                     # SCP generators
└── .claude_reports/                    # Documentation (gitignored)
    └── docs_code/                      # Code analysis docs
```

## Execution Examples

```bash
# Training (pretrain stage)
python run.py --model TF_Restormer --engine_mode train --config baseline.yaml

# Evaluation on test set
python run.py --model TF_Restormer --engine_mode eval --config baseline.yaml

# Inference (save enhanced WAVs)
python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml

# Folder-based inference
python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml \
  --input_dir /path/to/wavs --dump_path /path/to/output
```

## Coding Rules

- Config-driven: all hyperparameters, paths, and flags live in YAML
- Environment variables in paths use `${VAR_NAME}` syntax, resolved from `.env`
- RIR paths use aliases (e.g., `DNS_48K`) mapped in `.env`
- `testset_key` supports both string and list (sequential evaluation)
- `noisy_suffix` filters multi-channel files by suffix (e.g., `_ch1`)
- Engine selection is automatic based on `engine_mode` + optional `input_dir`
- Checkpoints saved under `models/TF_Restormer/log/log_{train_phase}_{dataset_phase}_{config}/weights/`
