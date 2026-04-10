# TF-Restormer for Speech Enhancement

Time-Frequency domain Restormer for speech enhancement.
[[Paper (arXiv)]](https://arxiv.org/pdf/2509.21003) [[Demo]](https://tf-restormer.github.io/demo/)

---

## Installation

**Requirements**: Python 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repository-url>
cd TF_Restormer_release

# CUDA 12.4 — inference only
uv sync --extra cu124

# CUDA 12.4 — full training + offline model
uv sync --extra cu124 --extra train

# CUDA 12.4 — full training + offline + streaming (Mamba) model
uv sync --extra cu124 --extra train --extra mamba

# CUDA 12.6 variants
uv sync --extra cu126
uv sync --extra cu126 --extra train --extra mamba

# CPU-only
uv sync --extra cpu
```

> The `--extra mamba` flag installs `mamba-ssm` and `causal-conv1d`, required only for the streaming (online) model. Build requires GCC/G++.

---

## Quick Start — Library API

```python
from tf_restormer import SEInference

# Load a pretrained model
model = SEInference.from_pretrained(
    config="baseline.yaml",
    checkpoint_path="path/to/checkpoints/baseline/",
    device="cuda",
)

# Enhance a file
result = model.process_file("noisy.wav", output_path="enhanced.wav")
# result["waveform"]    -> (1, L_out) tensor at 48 kHz
# result["sample_rate"] -> 48000

# Enhance a waveform tensor directly
import torch
waveform = torch.randn(1, 16000)     # (1, L) at 16 kHz input
result = model.process_waveform(waveform)
```

See `library_examples/` for complete runnable scripts.

---

## CLI Usage

### Prerequisites

Before training, set `db_root` and `rir_dir` in
`tf_restormer/models/TF_Restormer/configs/baseline.yaml` under the `to48k:` section:

```yaml
dataset:
    to48k:
        db_root: /path/to/your/dataset   # e.g. /home/DB/VCTK
        rir_dir: /path/to/DNS_RIR_48k    # e.g. /home/DB/DNS_RIR_48k
```

### Commands

```bash
# Train
uv run python run.py --model TF_Restormer --engine_mode train --config baseline.yaml

# Inference — single file
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml \
    --input noisy.wav --output enhanced/

# Inference — full test set (paths from config)
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml

# Evaluation — compute metrics on test set
uv run python run.py --model TF_Restormer --engine_mode eval --config baseline.yaml
```

Available configs: `baseline.yaml` (offline), `streaming.yaml` (online/Mamba).

---

## Checkpoint Management

```bash
# Export the latest checkpoint to tf_restormer/checkpoints/baseline/
python tf_restormer/export.py --config baseline.yaml

# Export and upload to Hugging Face Hub
python tf_restormer/export.py --config baseline.yaml --upload --repo-id shinuh/tf-restormer-baseline

# Upload all locally exported checkpoints
python tf_restormer/export.py --upload-all

# Download a checkpoint from Hugging Face Hub
python tf_restormer/export.py --download --config baseline.yaml --repo-id shinuh/tf-restormer-baseline
```

Requires `uv sync --extra hub` for Hugging Face upload/download.

---

## Examples

| File | Description |
|------|-------------|
| `library_examples/basic_inference.py` | Load a model and enhance a single file |
| `library_examples/batch_inference.py` | Enhance all `.wav` files in a directory |
| `library_examples/streaming_inference.py` | Chunk-by-chunk streaming (requires `--extra mamba`) |
| `library_examples/config_override.py` | Override config values at load time; HF Hub loading |
| `library_examples/eval_metrics.py` | Compute PESQ/STOI/DNSMOS/NISQA independently |

---

## Project Structure

```
TF_Restormer_release/
├── run.py                          # CLI entry point (train / infer / eval)
├── pyproject.toml                  # Package metadata and dependencies
│
├── tf_restormer/                   # Installable Python package
│   ├── __init__.py                 # Public API: SEInference, InferenceSession
│   ├── inference.py                # SEInference and InferenceSession classes
│   ├── export.py                   # Checkpoint export / HF Hub upload-download
│   ├── _config.py                  # Config loading helpers
│   │
│   ├── models/
│   │   └── TF_Restormer/
│   │       ├── model.py            # Model architecture entry point
│   │       ├── engine.py           # Training engine
│   │       ├── engine_infer.py     # Inference engine
│   │       ├── engine_eval.py      # Evaluation engine
│   │       ├── dataset.py          # Dataset and dataloader
│   │       ├── loss.py             # Loss functions
│   │       ├── main.py             # Train/eval orchestrator
│   │       ├── main_infer.py       # Inference orchestrator
│   │       ├── configs/
│   │       │   ├── baseline.yaml   # Offline model config
│   │       │   ├── streaming.yaml  # Online (Mamba) model config
│   │       │   └── testsets.yaml   # Test set definitions
│   │       └── modules/
│   │           ├── network.py      # Restormer network blocks
│   │           ├── module.py       # Sub-modules (attention, FFN, etc.)
│   │           └── msstftd.py      # Multi-scale STFT discriminator
│   │
│   └── utils/
│       ├── util_engine.py          # Checkpoint load/save helpers
│       ├── util_dataset.py         # Dataset utilities
│       ├── util_stft.py            # STFT helpers
│       ├── util_system.py          # System/logging utilities
│       ├── util_writer.py          # TensorBoard writer
│       ├── metrics/                # PESQ, STOI, DNSMOS, NISQA, ASR-WER
│       └── pos_embed.py            # Positional embedding utilities
│
├── library_examples/               # Runnable API usage examples
│   ├── basic_inference.py
│   ├── batch_inference.py
│   ├── streaming_inference.py
│   ├── config_override.py
│   └── eval_metrics.py
│
└── data/
    ├── create_scp/                 # Scripts to generate SCP file lists
    └── scp/                        # Generated SCP files (gitignored)
```
