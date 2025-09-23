# TF_Restormer

Time-Frequency domain Restormer for speech restoration

## Installation

### Prerequisites
- Python 3.12 (required)
- CUDA 12.4 or compatible version
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- GCC/G++ compiler for building native extensions

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TF_Restormer
```

2. Configure environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to set your CUDA path and dataset paths
# Example CUDA configuration:
# CUDA_HOME=/path/to/cuda-12.4
```

3. Create virtual environment and install dependencies:
```bash
# Using uv (recommended)
uv sync

# If some packages fail to build, install with proper compiler flags:
source .env
export CFLAGS="-O3" CXXFLAGS="-O3"
uv add --no-build-isolation-package <package-name> <package-name>
```

### Troubleshooting Installation

If you encounter build errors with C/C++ extensions (pysptk, pyworld, causal-conv1d, mamba-ssm):

```bash
# Load environment variables
source .env

# Set compiler flags to avoid debug flags issue
export CC=gcc
export CXX=g++
export CFLAGS="-O3"
export CXXFLAGS="-O3"

# Install packages one by one if needed
uv add pysptk --no-build-isolation-package pysptk
uv add pyworld --no-build-isolation-package pyworld
uv add causal-conv1d --no-build-isolation-package causal-conv1d
uv add mamba-ssm --no-build-isolation-package mamba-ssm
```

### Verify Installation

```bash
# Check if all packages are installed
uv pip list | grep -E "torch|pysptk|pyworld|causal-conv1d|mamba-ssm"

# Test PyTorch CUDA availability
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Training Guide

### Prerequisites for Training

1. **Dataset Preparation**
   - Download required datasets (LibriTTS-R, DNS-Challenge, etc.)
   - Update dataset paths in `.env` file

2. **Environment Configuration**
   ```bash
   # Copy and configure environment file
   cp .env.example .env
   
   # Edit .env to set your dataset paths:
   # DNS_DB_ROOT=/path/to/DNS-Challenge-16kHz
   # DNS_RIR_PATH=datasets_fullband/impulse_responses
   # LIBRI_TTS_R_DB_ROOT=/path/to/LibriTTS_R
   # ... (other dataset paths)
   ```

### Data Preparation

1. **Create SCP Files**
   
   SCP (Script) files are text files that map audio file identifiers to their full paths. They are required for the dataloader to find the audio files.

   ```bash
   # Create all necessary SCP files for LibriTTS-R dataset
   uv run data/create_scp/create_scp_all_libriTTS_R.py
   
   # This will create:
   # - data/scp/scp_LibriTTS_R/tr_s.scp (training speaker files)
   # - data/scp/scp_LibriTTS_R/cv_s.scp (validation speaker files)  
   # - data/scp/scp_LibriTTS_R/tr_n.scp (noise files)
   ```

2. **Verify SCP Files**
   ```bash
   # Check if SCP files are created properly
   wc -l data/scp/scp_LibriTTS_R/*.scp
   
   # Expected output (example):
   # 354729 data/scp/scp_LibriTTS_R/tr_s.scp
   # 10349 data/scp/scp_LibriTTS_R/cv_s.scp
   # 58454 data/scp/scp_LibriTTS_R/tr_n.scp
   ```

### Model Training

1. **Configuration**
   
   Each model version has its own configuration file:
   - `models/TF_Restormer/configs.yaml`
   
   Key configuration parameters:
   - `train_phase`: Training phase (`pretrain`, `adversarial`)
   - `dataset_phase`: Dataset configuration (`to48k`)
   - `batch_size`: Batch size for training
   - `max_epoch`: Maximum epochs for each phase

2. **Start Training**
   ```bash
   # Train TF_Restormer baseline(offline) model
   uv run run.py --model TF_Restormer --engine_mode train --config baseline.yaml

   # Train TF_Restormer streaming(online) model
   uv run run.py --model TF_Restormer --engine_mode train --config streaming.yaml
   ```

3. **Training Phases**
   
   The training consists of three phases:
   - **Pretrain Phase**: Basic model training with enhancement losses
   - **Adversarial Phase**: GAN training with discriminator

4. **Monitor Training**
   ```bash
   # Check training logs
   tail -f models/TF_Restormer_v3/log/system_log.log
   
   # Monitor GPU usage
   watch -n 1 nvidia-smi
   ```

### Common Issues and Solutions

1. **Missing Dependencies**
   ```bash
   # If pesq module is missing
   uv add pesq
   
   # If discrete-speech-metrics fails to install
   # The training will continue with a warning, but some metrics will be disabled
   ```

2. **Missing SCP Files**
   ```bash
   # Error: FileNotFoundError: File not found: data/scp/scp_LibriTTS_R/tr_n.scp
   # Solution: Run the SCP creation script
   uv run data/create_scp/create_scp_all_libriTTS_R.py
   ```

3. **Dataset Path Issues**
   ```bash
   # Error: DNS noise path does not exist
   # Solution: Update paths in .env file to match your system
   # Then regenerate SCP files
   ```

4. **RIR (Room Impulse Response) Path Issues**
   ```bash
   # RIR paths are managed via aliases in configs.yaml and actual paths in .env:
   # In configs.yaml: rir: "DNS_16K" or rir: "DNS_48K"
   # In .env file: 
   # RIR_DNS_16K=/path/to/16khz/impulse_responses
   # RIR_DNS_48K=/path/to/48khz/impulse_responses
   ```

### Monitoring Training with TensorBoard

```bash
# Run TensorBoard to monitor training progress
# Note: Due to NumPy version compatibility issues, use the following command:
uv run python -m tensorboard.main --logdir=models/model_name/log

# TensorBoard will be available at http://localhost:6006/
# You may see a warning about TensorFlow not being found, but TensorBoard will still work
```

### Inference (e.g. baseline.yaml)

```bash
# Run inference on a test set and save on default dir
uv run run.py --model TF_Restormer --engine_mode infer --config baseline.yaml


# if specfiying dump dir
uv run run.py --model TF_Restormer --engine_mode infer --config baseline.yaml --dump_path /path/to/dump

```

### Testing

```bash
# Run evaluation on test dataset
uv run run.py --model TF_Restormer --engine_mode test
```

## Development

### Project Structure
```
TF_Restormer/
├── data/
│   ├── create_scp/     # Scripts to create SCP files
│   └── scp/             # Generated SCP files
├── models/
│   └── TF_Restormer/ # baseline model
├── utils/               # Utility functions
├── .env                 # Environment configuration (create from .env.example)
├── pyproject.toml       # Project dependencies
└── run.py              # Main entry point
```

### Adding New Datasets

1. Add dataset paths to `.env`
2. Create SCP generation script in `data/create_scp/`
3. Update model configuration in `configs.yaml`
4. Run SCP generation script before training
