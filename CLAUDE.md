# CLAUDE.md — TF-Restormer Project Guide

Time-Frequency domain Restormer speech enhancement package.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [01 Architecture Overview](.claude_reports/docs_code/01_architecture_overview.md) | Model architecture, data flow, STFT pipeline |
| [02 Model / Network](.claude_reports/docs_code/02_model_network.md) | Restormer blocks, attention, FFN modules |
| [03 Engine](.claude_reports/docs_code/03_engine.md) | Training, inference, and evaluation engines |
| [04 Dataset](.claude_reports/docs_code/04_dataset.md) | Dataset class, SCP file format, augmentation |
| [05 Loss](.claude_reports/docs_code/05_loss.md) | Loss functions, GAN discriminator |
| [06 Utils](.claude_reports/docs_code/06_utils.md) | Metrics, STFT helpers, checkpoint utilities |
| [07 Config Reference](.claude_reports/docs_code/07_config_reference.md) | Full YAML config field reference |

---

## Project Structure

```
TF_Restormer_release/
├── run.py                          # CLI entry point (train / infer / eval)
├── pyproject.toml                  # Package metadata and extras
├── CLAUDE.md                       # This file
│
├── tf_restormer/                   # Installable Python package
│   ├── __init__.py                 # Public API: SEInference, InferenceSession (lazy import)
│   ├── inference.py                # SEInference.from_pretrained(), process_file(), process_waveform()
│   ├── export.py                   # export_checkpoint(), upload/download HF Hub CLI
│   ├── _config.py                  # load_config(), resolve_config(), _VARIANT_MAP
│   │
│   ├── models/
│   │   └── TF_Restormer/
│   │       ├── model.py            # Model class — entry point for architecture
│   │       ├── engine.py           # Training engine (GAN, pretrain phases)
│   │       ├── engine_infer.py     # EngineInfer — single-file / batch inference
│   │       ├── engine_eval.py      # Evaluation engine, metric aggregation
│   │       ├── dataset.py          # TFDataset, SCP loading, online augmentation
│   │       ├── loss.py             # SI-SDR, spectral, GAN losses
│   │       ├── main.py             # Train/eval orchestrator (called by run.py)
│   │       ├── main_infer.py       # Inference orchestrator (called by run.py)
│   │       └── configs/
│   │           ├── baseline.yaml   # Offline model
│   │           ├── streaming.yaml  # Online model (Mamba)
│   │           └── testsets.yaml   # Test set path definitions
│   │       └── modules/
│   │           ├── network.py      # Full Restormer network
│   │           ├── module.py       # TF attention, gated FFN, patch embed
│   │           └── msstftd.py      # Multi-scale STFT discriminator
│   │
│   └── utils/
│       ├── util_engine.py          # _find_latest_checkpoint(), _fix_compiled_state_dict()
│       ├── util_dataset.py         # Audio I/O, resampling, SCP utilities
│       ├── util_stft.py            # STFT / iSTFT wrappers
│       ├── util_system.py          # Logging setup, reproducibility seeds
│       ├── util_writer.py          # TensorBoard SummaryWriter wrapper
│       ├── pos_embed.py            # 2-D positional embedding
│       ├── decorators.py           # @timer, @deprecated, etc.
│       └── metrics/
│           ├── intrusive.py        # PESQ, STOI, SDR, LSD, MCD, composite
│           ├── nonintrusive.py     # DNSMOS, NISQA
│           ├── neural.py           # Deep metric wrappers
│           ├── semantic.py         # UTMOS, SHEET
│           └── asr.py              # ASR-WER (Whisper / wav2vec2)
│
├── library_examples/               # Standalone runnable usage examples
│   ├── basic_inference.py          # Minimal single-file enhancement
│   ├── batch_inference.py          # Directory-level batch processing
│   ├── streaming_inference.py      # Chunk-by-chunk streaming (Mamba model)
│   ├── config_override.py          # Config patterns + HF Hub loading
│   └── eval_metrics.py             # Metric computation standalone
│
└── data/
    ├── create_scp/                 # SCP generation scripts
    └── scp/                        # Generated SCP files (gitignored)
```

---

## Execution

### Install

```bash
# CUDA 12.4, full training + streaming model
uv sync --extra cu124 --extra train --extra mamba

# Inference only (no training deps)
uv sync --extra cu124

# CPU-only
uv sync --extra cpu
```

### CLI

```bash
# Train
uv run python run.py --model TF_Restormer --engine_mode train --config baseline.yaml

# Inference — single file
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml \
    --input noisy.wav --output enhanced/

# Inference — full test set
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml

# Evaluation — compute metrics
uv run python run.py --model TF_Restormer --engine_mode eval --config baseline.yaml
```

Set `db_root` and `rir_dir` in `tf_restormer/models/TF_Restormer/configs/baseline.yaml`
before running training (fields are `null` by default).

### Library API

```python
from tf_restormer import SEInference

model = SEInference.from_pretrained(
    config="baseline.yaml",          # or absolute path, or dict
    checkpoint_path="path/to/ckpt/", # directory with model.pt, or .pt file
    device="cuda",
)

result = model.process_file("noisy.wav", output_path="enhanced.wav")
# result["waveform"]    -> (1, L_out) at 48 kHz
# result["sample_rate"] -> 48000
```

### Checkpoint Management

```bash
python tf_restormer/export.py --config baseline.yaml                          # export
python tf_restormer/export.py --config baseline.yaml --upload --repo-id owner/repo  # upload
python tf_restormer/export.py --download --config baseline.yaml --repo-id owner/repo  # download
```

---

## Key Architecture Facts

- **Input**: 16 kHz waveform → upsampled to 48 kHz inside the pipeline.
- **STFT**: 960-point FFT (20 ms frame), 480-point hop (10 ms).
- **Backbone**: Restormer encoder-decoder in TF domain; channel attention + gated FFN.
- **Training phases**: `pretrain` (enhancement losses only) → `adversarial` (+ GAN).
- **Streaming model**: replaces self-attention with causal Mamba SSM; requires `--extra mamba`.
- **Output**: 48 kHz enhanced waveform.

---

## Behavioral Guidelines

### Code Style

- **Docstrings and comments**: English only.
- **User-facing explanations (to the developer)**: Korean is acceptable.
- **Type hints**: use for all public functions and class methods.
- **Imports**: standard library → third-party → internal (`tf_restormer.*`).

### Architecture Rules

- Do not bypass `load_config()` / `resolve_config()` for config loading.
- `SEInference.from_pretrained()` is the single public entry point for inference; do not call `EngineInfer` directly from application code.
- Keep `tf_restormer/__init__.py` lazy — do not import torch at module load time.
- Training log directories (`tf_restormer/models/*/log/`) are gitignored; use `export.py` to publish checkpoints.

### Commit Rules

- Prefix: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`.
- Keep commits small and focused. One logical change per commit.
- Do not commit `CLAUDE.md` to public repositories without explicit user approval.

### Environment

- **Primary tool**: `uv` (not conda, not plain pip for dev installs).
- **Python version**: 3.12 (pinned in `.python-version`).
- **No `.env` file**: dataset paths go directly in the YAML config (`db_root`, `rir_dir`).
