# TF-Restormer for Speech Enhancement

Time-Frequency domain Restormer for speech enhancement.
[[Paper (arXiv)]](https://arxiv.org/pdf/2509.21003) [[Demo]](https://tf-restormer.github.io/demo/)

## Overview

TF-Restormer implements a Time-Frequency domain Restormer architecture for single-channel speech enhancement. The model operates in the STFT domain with a Restormer encoder-decoder backbone featuring channel attention and gated feed-forward networks. It accepts wideband (16 kHz) input and produces fullband (48 kHz) enhanced output. The framework supports two model variants — an offline model (TF-Locoformer) and an online streaming model (Mamba SSM) — and provides both a training/evaluation CLI and a clean Python library API for inference.

## Installation

### uv (recommended)

[uv](https://docs.astral.sh/uv/) for dependency management with conflict-safe PyTorch index routing.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc                                           # apply to current shell

# 2. Install dependencies
uv sync --extra cu124                              # CUDA 12.4 inference only
uv sync --extra cu124 --extra train                # CUDA 12.4 + training
uv sync --extra cu124 --extra train --extra mamba  # CUDA 12.4 + training + streaming (Mamba)
uv sync --extra cu126 --extra train --extra mamba  # CUDA 12.6 variant
uv sync --extra cpu                                # CPU-only inference

# 3. Activate
source .venv/bin/activate    # or prefix commands with: uv run python run.py ...
```

> **NOTE**: The `--extra mamba` flag installs `mamba-ssm` and `causal-conv1d`, required only for the streaming (online) model. Build requires GCC/G++. Do not use `uv sync --extra train` alone — always combine with an accelerator extra (`cu124`/`cu126`/`cpu`).

## Project Structure

```
TF_Restormer_release/
  run.py                            # CLI entry point
  tf_restormer/
    inference.py                    # Public API (SEInference, InferenceSession)
    export.py                       # Checkpoint export and HF Hub upload utilities
    _config.py                      # Config loading helpers
    models/
      TF_Restormer/
        model.py / engine.py        # Model definition and train/test loops
        engine_infer.py             # Tensor-in / tensor-out inference engine
        engine_eval.py              # Evaluation engine with metric aggregation
        modules/                    # network.py (TF blocks, attention) + module.py
        configs/                    # Per-experiment YAML config files
    utils/                          # STFT, metrics, checkpoints, dataset utilities
  library_examples/                 # Library API usage examples
  data/                             # SCP generation scripts
```

## Data Preparation

Training requires SCP (script) files that map utterance keys to audio file paths.

```bash
# Generate SCP files for specific datasets
python data/create_scp/create_scp_VCTK.py
python data/create_scp/create_scp_libriTTS_R.py
python data/create_scp/create_scp_noise.py
```

Before training, set `db_root` and `rir_dir` in
`tf_restormer/models/TF_Restormer/configs/baseline.yaml`:

```yaml
dataset:
    to48k:
        db_root: /path/to/your/dataset   # e.g. /home/DB/VCTK
        rir_dir: /path/to/DNS_RIR_48k    # e.g. /home/DB/DNS_RIR_48k
```

Generated SCP files are saved to `data/scp/` and referenced by training configs.

## Usage

### Local Training / Inference / Evaluation (CLI)

Use `run.py` for local model training, evaluation, and file-based inference.

```bash
# Training
uv run python run.py --model TF_Restormer --engine_mode train --config baseline.yaml

# Inference — single file
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml \
    --input noisy.wav --output enhanced/

# Inference — full test set (paths from config)
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml

# Evaluation — compute metrics on test set
uv run python run.py --model TF_Restormer --engine_mode eval --config baseline.yaml
```

CLI flags: `--model` (TF_Restormer), `--engine_mode` (train/infer/eval), `--config` (YAML filename), `--input`, `--output`.

Available configs: `baseline.yaml` (offline), `streaming.yaml` (online/Mamba).

#### Checkpoint Management

Export, upload, and download checkpoints via `tf_restormer/export.py`:

```bash
# Export a trained checkpoint (strip optimizer state for deployment)
python tf_restormer/export.py --config baseline.yaml

# Upload to Hugging Face Hub
python tf_restormer/export.py --config baseline.yaml --upload

# Upload with explicit repo ID
python tf_restormer/export.py --config baseline.yaml --upload --repo-id shinuh/tf-restormer-baseline

# Upload all locally exported checkpoints
python tf_restormer/export.py --upload-all

# Download from Hugging Face Hub
python tf_restormer/export.py --download --repo-id shinuh/tf-restormer-baseline
```

Requires `uv sync --extra hub` for Hugging Face upload/download.

#### Pretrained Models

Available on [Hugging Face Hub](https://huggingface.co/shinuh):

| Model | Config | Repo ID | Description |
|---|---|---|---|
| Offline | `baseline.yaml` | `shinuh/tf-restormer-baseline` | TF-Locoformer, 48 kHz output |
| Online | `streaming.yaml` | `shinuh/tf-restormer-streaming` | Mamba SSM, causal streaming |

### Library API (Python)

For programmatic inference — load a model and process audio in Python code.
Supports local checkpoints and Hugging Face Hub downloads.

#### Model Loading

```python
from tf_restormer import SEInference

# Load from local checkpoint
model = SEInference.from_pretrained(
    config="baseline.yaml",
    checkpoint_path="path/to/checkpoints/baseline/",
    device="cuda",
)

# Or load from Hugging Face Hub (requires: uv sync --extra hub)
model = SEInference.from_pretrained(
    checkpoint_path="shinuh/tf-restormer-baseline",
    device="cuda",
)
```

#### Level 1: File I/O (`process_file`)

```python
result = model.process_file("noisy.wav", output_path="enhanced.wav")
# result["waveform"]    -> (1, L_out) tensor at 48 kHz
# result["sample_rate"] -> 48000
```

#### Level 2: Waveform Tensor (`process_waveform`)

```python
import torch
waveform = torch.randn(1, 16000)     # (1, L) at 16 kHz input
result = model.process_waveform(waveform)
# result["waveform"]    -> (1, L_out) tensor at 48 kHz
```

### Examples

See `library_examples/` for ready-to-run scripts:

| Script | Description |
|---|---|
| `basic_inference.py` | Load a model and enhance a single file |
| `batch_inference.py` | Enhance all `.wav` files in a directory |
| `streaming_inference.py` | Chunk-by-chunk streaming (requires `--extra mamba`) |
| `config_override.py` | Override config values at load time; HF Hub loading |
| `eval_metrics.py` | Compute PESQ/STOI/DNSMOS/NISQA independently |

## Architecture

```
Waveform (16 kHz)
    |
    v
Resample (-> 48 kHz)  ->  STFT (960-pt FFT, 480-pt hop)
    |
    v
TF-Restormer Encoder (channel attention + gated FFN)
    |
    v
Frequency Upsampler
    |
    v
TF-Restormer Decoder (channel attention + gated FFN)
    |
    v
Spectral Estimator  ->  iSTFT  ->  Enhanced Waveform (48 kHz)
```

Key components: Conv1d-based differentiable STFT/iSTFT (`util_stft.py`), Restormer encoder-decoder with TF-domain attention and Rotary Position Embedding, frequency upsampler between stages, and two-phase training (pretrain + adversarial with multi-scale STFT discriminator).

The streaming variant replaces self-attention with causal Mamba SSM for online processing.

## Citation

If you use TF-Restormer in your research, please cite:

```bibtex
@article{tfrestormer2025,
  title   = {TF-Restormer: Time-Frequency Domain Restormer for Speech Enhancement},
  author  = {},
  journal = {arXiv preprint arXiv:2509.21003},
  year    = {2025},
}
```
