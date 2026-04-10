"""
tf_restormer.utils.metrics.neural
==================================
Neural MOS metrics: UTMOS (torch.hub SpeechMOS) and WVMOS (Wav2Vec2MOS).

Both functions use _model_cache for GPU-based model caching and lazy imports
so that `transformers` / `torch.hub` are not required at import time.
"""

from __future__ import annotations

import numpy as np

from tf_restormer.utils.metrics import _model_cache


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_tensor(x, device: str):
    """Convert np.ndarray or torch.Tensor to a float32 1-D torch.Tensor on device."""
    import torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(torch.float32)
    if x.dim() == 0:
        x = x.unsqueeze(0)
    return x.to(device)


# ---------------------------------------------------------------------------
# UTMOS
# ---------------------------------------------------------------------------

def compute_utmos(
    estim,
    target=None,
    fs: int = 16000,
    *,
    device: str = "cuda",
    cache=None,
    **kwargs,
) -> float:
    """Compute UTMOS22 strong MOS score for a single waveform.

    Parameters
    ----------
    estim:  Enhanced/estimated waveform (torch.Tensor 1-D or np.ndarray).
    target: Ignored (UTMOS is non-intrusive). Accepted for API uniformity.
    fs:     Sampling rate in Hz (default 16000).
    device: Torch device string (default ``"cuda"``).
    cache:  _ModelCache instance. Defaults to the module-level ``_model_cache``.

    Returns
    -------
    float — mean UTMOS score across frames.
    """
    import torch

    cache = cache or _model_cache

    def _factory(dev):
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        model.to(dev)
        model.eval()
        return model

    model = cache.get_or_create("utmos22_strong", device, _factory)

    wav = _to_tensor(estim, device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # (1, T)

    with torch.no_grad():
        scores = model(wav, sr=fs)  # (1,) or (B,)

    return scores.mean().item()


# ---------------------------------------------------------------------------
# WVMOS
# ---------------------------------------------------------------------------

def _build_wvmos(device: str):
    """Factory: construct Wav2Vec2MOS and load pretrained weights."""
    import os
    import urllib.request
    from collections import OrderedDict

    import torch
    import torch.nn as nn

    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
    except ImportError:
        raise ImportError(
            "WVMOS requires 'transformers'. "
            "Install with: pip install tf-restormer[metrics-neural]"
        )

    # --- checkpoint download ---
    path = os.path.join(os.path.expanduser("~"), ".cache/wv_mos/wv_mos.ckpt")
    if not os.path.exists(path):
        print("Downloading the checkpoint for WV-MOS")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
            path,
        )
        print(f"Weights downloaded in: {path}  Size: {os.path.getsize(path)}")

    # --- model construction (mirrors Wav2Vec2MOS.__init__) ---
    class _Wav2Vec2MOS(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.dense = nn.Sequential(
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            x = self.encoder(x)["last_hidden_state"]  # (B, T, D)
            x = self.dense(x)                          # (B, T, 1)
            x = x.mean(dim=[1, 2], keepdims=True)      # (B, 1, 1)
            return x

    def _extract_prefix(prefix, weights):
        result = OrderedDict()
        for key in weights:
            if key.startswith(prefix):
                result[key[len(prefix):]] = weights[key]
        return result

    model = _Wav2Vec2MOS()
    state = torch.load(path, weights_only=False, map_location=device)["state_dict"]
    model.load_state_dict(_extract_prefix("model.", state))

    model.to(device)
    # Freeze encoder weights for eval (mirrors original freeze=True default)
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.eval()

    # Attach processor as an attribute for use in compute_wvmos
    model._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    return model


def compute_wvmos(
    estim,
    target=None,
    fs: int = 16000,
    *,
    device: str = "cuda",
    cache=None,
    **kwargs,
) -> float:
    """Compute WV-MOS (Wav2Vec2-based MOS) for a single waveform.

    Parameters
    ----------
    estim:  Enhanced/estimated waveform (torch.Tensor 1-D or np.ndarray).
    target: Ignored (WVMOS is non-intrusive). Accepted for API uniformity.
    fs:     Sampling rate in Hz. WV-MOS model expects 16 kHz.
    device: Torch device string (default ``"cuda"``).
    cache:  _ModelCache instance. Defaults to the module-level ``_model_cache``.

    Returns
    -------
    float — mean WV-MOS score.
    """
    import torch

    cache = cache or _model_cache
    model = cache.get_or_create("wvmos", device, _build_wvmos)

    # Convert input to numpy for processor, then to tensor on device
    if isinstance(estim, torch.Tensor):
        signal = estim.detach().cpu().float().numpy()
    else:
        signal = np.asarray(estim, dtype=np.float32)

    if signal.ndim > 1:
        signal = signal.squeeze()

    x = model._processor(
        signal,
        return_tensors="pt",
        padding=True,
        sampling_rate=16000,
    ).input_values.to(device)

    with torch.no_grad():
        result = model(x).mean()

    return result.item()
