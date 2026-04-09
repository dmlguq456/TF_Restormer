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
| `tf_restormer/models/TF_Restormer/main.py` | 01, 03 |
| `tf_restormer/models/TF_Restormer/model.py` | 02 |
| `tf_restormer/models/TF_Restormer/modules/module.py` | 02 |
| `tf_restormer/models/TF_Restormer/modules/network.py` | 02 |
| `tf_restormer/models/TF_Restormer/modules/msstftd.py` | 02 |
| `tf_restormer/models/TF_Restormer/engine.py` | 03 |
| `tf_restormer/models/TF_Restormer/dataset.py` | 04 |
| `tf_restormer/models/TF_Restormer/loss.py` | 05 |
| `tf_restormer/utils/util_stft.py` | 06 |
| `tf_restormer/utils/util_dataset.py` | 06 |
| `tf_restormer/utils/util_engine.py` | 06 |
| `tf_restormer/utils/util_metric.py` | 06 |
| `tf_restormer/utils/util_wvmos.py` | 06 |
| `tf_restormer/utils/util_dnsmos.py` | 06 |
| `tf_restormer/utils/util_nisqa.py` | 06 |
| `tf_restormer/utils/util_composite.py` | 06 |
| `tf_restormer/utils/util_mcd.py` | 06 |
| `tf_restormer/utils/util_pesq.py` | 06 |
| `tf_restormer/utils/util_writer.py` | 06 |
| `tf_restormer/utils/util_system.py` | 06 |
| `tf_restormer/utils/util_speechbertscore.py` | 06 |
| `tf_restormer/utils/util_speechbleu.py` | 06 |
| `tf_restormer/utils/util_speechtokendistance.py` | 06 |
| `tf_restormer/utils/util_sBERTscore.py` | 06 |
| `tf_restormer/utils/pos_embed.py` | 06 |
| `tf_restormer/utils/decorators.py` | 06 |
| `tf_restormer/utils/ASR_whisper.py` | 06 |
| `tf_restormer/utils/ASR_w2v.py` | 06 |
| `tf_restormer/utils/NISQA_models/NISQA_lib.py` | 06 |
| `tf_restormer/utils/test_nisqa.py` | 06 |

## Project Structure

```
TF_Restormer_release/
├── run.py                              # CLI entry point
├── CLAUDE.md
├── pyproject.toml                      # Package config (tf_restormer, editable install)
├── tf_restormer/                       # Python package root
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── TF_Restormer/
│   │       ├── __init__.py
│   │       ├── main.py                 # Orchestrator
│   │       ├── model.py                # Model_Enhance network
│   │       ├── engine.py               # Train/Eval/Infer engines
│   │       ├── dataset.py              # Data loading & synthesis
│   │       ├── loss.py                 # Loss functions
│   │       ├── modules/
│   │       │   ├── __init__.py
│   │       │   ├── module.py           # Encoder, Decoder, Upsampler
│   │       │   ├── network.py          # Attention, Transformer, Mamba
│   │       │   └── msstftd.py          # Multi-scale STFT discriminator
│   │       └── configs/
│   │           ├── baseline.yaml       # Main config
│   │           └── .legacy/            # Archived configs (gitignored)
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       ├── util_stft.py                # STFT/iSTFT
│       ├── util_metric.py              # PESQ, STOI, SDR, LSD, MCD
│       ├── util_engine.py              # Checkpointing, schedulers
│       ├── util_dataset.py             # SCP parsing, audio processing
│       ├── util_wvmos.py, util_dnsmos.py   # MOS metrics
│       ├── util_nisqa.py, util_composite.py
│       ├── util_writer.py              # TensorBoard logging
│       ├── util_system.py              # YAML, WandB
│       ├── ASR_whisper.py, ASR_w2v.py  # ASR utilities
│       ├── NISQA_models/               # NISQA checkpoints
│       ├── dnsmos_models/              # DNSMOS ONNX models
│       └── km/                         # K-means bins (SpeechBLEU/TokenDistance)
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
- Checkpoints saved under `tf_restormer/models/TF_Restormer/log/log_{train_phase}_{dataset_phase}_{config}/weights/`
- Package installed as editable install: `pip install -e .` (required before running `run.py`)
- All imports use `tf_restormer.*` absolute paths; intra-subpackage references use relative imports (`.xxx`)
