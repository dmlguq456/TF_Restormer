# TF-Restormer for Speech Enhancement

Time-Frequency domain Restormer for speech enhancement.
[[Paper (arXiv)]](https://arxiv.org/pdf/2509.21003) [[Demo]](https://tf-restormer.github.io/demo/)

## Overview

TF-Restormer implements a Time-Frequency domain Restormer architecture for single-channel speech enhancement. The model operates in the STFT domain with a Restormer encoder-decoder backbone featuring channel attention and gated feed-forward networks. It accepts wideband (16 kHz) input and produces fullband (48 kHz) enhanced output. Two model variants are available:

- **Offline** (TF-Locoformer) — non-causal, higher quality
- **Online** (Mamba SSM) — causal streaming, low latency

## Installation

**Requirements**: Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
pip install tf-restormer
```

Or install from source:

```bash
git clone https://github.com/shinuh/TF-Restormer.git
cd TF-Restormer

# Inference only (CUDA 12.4)
uv sync --extra cu124

# With streaming model support (Mamba)
uv sync --extra cu124 --extra mamba

# CPU-only
uv sync --extra cpu
```

> The `--extra mamba` flag installs `mamba-ssm` and `causal-conv1d`, required only for the streaming (online) model.

## Quick Start

### Python API

```python
from tf_restormer import SEInference

# Load from Hugging Face Hub
model = SEInference.from_pretrained(
    checkpoint_path="shinuh/tf-restormer-baseline",
    device="cuda",
)

# Enhance a file
result = model.process_file("noisy.wav", output_path="enhanced.wav")
# result["waveform"]    -> (1, L) tensor at 48 kHz
# result["sample_rate"] -> 48000

# Or enhance a waveform tensor directly
import torch
waveform = torch.randn(1, 16000)  # (1, L) at 16 kHz
result = model.process_waveform(waveform)
```

### CLI

```bash
# Enhance a single file
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml \
    --input noisy.wav --output enhanced/

# Enhance a full test set
uv run python run.py --model TF_Restormer --engine_mode infer --config baseline.yaml

# Evaluate metrics (PESQ, STOI, DNSMOS, etc.)
uv run python run.py --model TF_Restormer --engine_mode eval --config baseline.yaml
```

## Pretrained Models

Pretrained checkpoints will be available on [Hugging Face Hub](https://huggingface.co/shinuh) soon.

| Model | Repo ID | Description |
|---|---|---|
| Offline | `shinuh/tf-restormer-baseline` | TF-Locoformer, 48 kHz output |
| Online | `shinuh/tf-restormer-streaming` | Mamba SSM, causal streaming |

```python
from tf_restormer import SEInference

# Offline model
model = SEInference.from_pretrained(checkpoint_path="shinuh/tf-restormer-baseline", device="cuda")

# Streaming model (requires --extra mamba)
model = SEInference.from_pretrained(checkpoint_path="shinuh/tf-restormer-streaming", device="cuda")
```

## Examples

See [`library_examples/`](library_examples/) for complete runnable scripts:

| Script | Description |
|---|---|
| `basic_inference.py` | Load a model and enhance a single file |
| `batch_inference.py` | Enhance all `.wav` files in a directory |
| `streaming_inference.py` | Chunk-by-chunk streaming inference |
| `config_override.py` | Override config values; HF Hub loading |
| `eval_metrics.py` | Compute PESQ/STOI/DNSMOS/NISQA standalone |

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

The model uses a Restormer encoder-decoder in the TF domain with Rotary Position Embedding and a frequency upsampler between stages. Training follows a two-phase approach: pretrain (enhancement losses) then adversarial (+ multi-scale STFT discriminator). The streaming variant replaces self-attention with causal Mamba SSM.

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

## License

[TBD]
