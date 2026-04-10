"""
tf_restormer.utils.metrics.nonintrusive
=========================================
Non-intrusive (reference-free) speech quality metrics.

All functions accept both ``torch.Tensor`` and ``np.ndarray`` as input and
convert internally. External dependencies are lazy-imported inside each
function.

Functions
---------
_run_dnsmos(wav, fs=16000, **kwargs) -> dict
    Shared private helper. Runs the DNSMOS ONNX model and returns the full
    result dict ``{'ovrl_mos', 'sig_mos', 'bak_mos', 'p808_mos'}``.
    The DNSMOS model object is cached via ``_model_cache`` from this package's
    ``__init__``.

compute_dnsmos(wav, fs=16000, **kwargs) -> float
    Overall MOS (ovrl_mos).

compute_dnsmos_sig(wav, fs=16000, **kwargs) -> float
    Signal MOS (sig_mos).

compute_dnsmos_bak(wav, fs=16000, **kwargs) -> float
    Background MOS (bak_mos).

compute_nisqa(wav, fs=16000, *, device='cpu', **kwargs) -> float
    NISQA MOS score.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Import the shared model cache from this package
from tf_restormer.utils.metrics import _model_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_1d(x) -> np.ndarray:
    """Convert torch.Tensor or np.ndarray to a 1-D float32 numpy array."""
    try:
        import torch
        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()
    except ImportError:
        pass
    arr = np.asarray(x, dtype=np.float32).squeeze()
    return arr


# ---------------------------------------------------------------------------
# DNSMOS — model class (migrated from util_dnsmos.py)
# ---------------------------------------------------------------------------

# Module-level constants (same as util_dnsmos.py)
_DNSMOS_SR = 16000
_DNSMOS_INPUT_LENGTH = 9.01


def _peak_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.max(np.abs(x)) + eps)


class _DNSMOSModel:
    """ONNX-based DNSMOS scorer. Migrated from ``util_dnsmos.DNSMOS``."""

    def __init__(self, primary_model_path: str, p808_model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "DNSMOS requires 'onnxruntime'. "
                "Install with: pip install tf-restormer[metrics-nonintrusive]"
            )
        self.primary_model_path = primary_model_path
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(
        self,
        audio: np.ndarray,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = 16000,
        to_db: bool = True,
    ) -> np.ndarray:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1,
            hop_length=hop_length, n_mels=n_mels,
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(
        self,
        sig: float,
        bak: float,
        ovr: float,
        is_personalized_MOS: bool,
    ):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101,    1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611,   0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def __call__(
        self,
        sample: np.ndarray,
        fs: int,
        is_personalized_MOS: bool,
    ) -> dict:
        clip_dict: dict = {}

        if isinstance(sample, np.ndarray):
            audio = sample
            if not ((audio >= -1).all() and (audio <= 1).all()):
                raise ValueError("np.ndarray values must be between -1 and 1.")
        elif isinstance(sample, str) and os.path.isfile(sample):
            import librosa
            audio, _ = librosa.load(sample, sr=fs)
            clip_dict['filename'] = sample
        else:
            raise ValueError(
                "Input must be a numpy array or a path to an audio file."
            )

        len_samples = int(_DNSMOS_INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - _DNSMOS_INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg: list[float] = []
        predicted_mos_bak_seg: list[float] = []
        predicted_mos_ovr_seg: list[float] = []
        predicted_p808_mos: list[float] = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples): int((idx + _DNSMOS_INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype('float32')[np.newaxis, :, :]

            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}

            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict['ovrl_mos'] = float(np.mean(predicted_mos_ovr_seg))
        clip_dict['sig_mos'] = float(np.mean(predicted_mos_sig_seg))
        clip_dict['bak_mos'] = float(np.mean(predicted_mos_bak_seg))
        clip_dict['p808_mos'] = float(np.mean(predicted_p808_mos))
        return clip_dict


# ---------------------------------------------------------------------------
# DNSMOS — shared private runner
# ---------------------------------------------------------------------------

def _run_dnsmos(
    wav,
    fs: int = 16000,
    model_type: str = "dnsmos",
    **kwargs,
) -> dict:
    """Run DNSMOS model and return the full result dict.

    Caches the DNSMOS model instance via ``_model_cache`` keyed by
    ``'dnsmos'`` + ``'cpu'`` (ONNX models are CPU-only).

    Parameters
    ----------
    wav:        Input waveform (1-D torch.Tensor or np.ndarray).
    fs:         Sampling rate of ``wav`` in Hz.
    model_type: ``'dnsmos'`` or ``'dnsmos_personalized'``.

    Returns
    -------
    dict with keys ``{'ovrl_mos', 'sig_mos', 'bak_mos', 'p808_mos'}``.
    """
    try:
        import onnxruntime  # noqa: F401 — raises ImportError if absent
    except ImportError:
        raise ImportError(
            "DNSMOS requires 'onnxruntime'. "
            "Install with: pip install tf-restormer[metrics-nonintrusive]"
        )
    import librosa

    # Resolve model paths relative to this file's parent (utils/)
    _utils_dir = Path(__file__).parent.parent  # .../tf_restormer/utils/
    p808_model_path = str(_utils_dir / "dnsmos_models" / "model_v8.onnx")

    is_personalized_eval = (model_type == "dnsmos_personalized")
    if is_personalized_eval:
        primary_model_path = str(_utils_dir / "pdnsmos_models" / "sig_bak_ovr.onnx")
    else:
        primary_model_path = str(_utils_dir / "dnsmos_models" / "sig_bak_ovr.onnx")

    # Factory builds (or re-uses) a _DNSMOSModel for this primary_model_path.
    # Cache key includes the primary_model_path so personalized vs standard
    # models are stored separately.
    cache_key = f"dnsmos:{primary_model_path}"

    def _factory(_device: str) -> _DNSMOSModel:
        return _DNSMOSModel(primary_model_path, p808_model_path)

    dnsmos_model: _DNSMOSModel = _model_cache.get_or_create(
        cache_key, 'cpu', _factory
    )

    # Convert input
    sample = _to_numpy_1d(wav)

    # Resample if needed
    if fs != _DNSMOS_SR:
        sample = librosa.resample(sample, orig_sr=fs, target_sr=_DNSMOS_SR)
        fs = _DNSMOS_SR

    sample = _peak_norm(sample)

    results = dnsmos_model(sample, fs, is_personalized_eval)
    return results


# ---------------------------------------------------------------------------
# DNSMOS — public functions
# ---------------------------------------------------------------------------

def compute_dnsmos(wav, fs: int = 16000, **kwargs) -> float:
    """DNSMOS overall MOS (ovrl_mos).

    Parameters
    ----------
    wav: Input waveform (1-D torch.Tensor or np.ndarray).
    fs:  Sampling rate in Hz.

    Returns
    -------
    float — overall MOS score.
    """
    return _run_dnsmos(wav, fs, **kwargs)['ovrl_mos']


def compute_dnsmos_sig(wav, fs: int = 16000, **kwargs) -> float:
    """DNSMOS signal quality MOS (sig_mos).

    Parameters
    ----------
    wav: Input waveform (1-D torch.Tensor or np.ndarray).
    fs:  Sampling rate in Hz.

    Returns
    -------
    float — signal MOS score.
    """
    return _run_dnsmos(wav, fs, **kwargs)['sig_mos']


def compute_dnsmos_bak(wav, fs: int = 16000, **kwargs) -> float:
    """DNSMOS background MOS (bak_mos).

    Parameters
    ----------
    wav: Input waveform (1-D torch.Tensor or np.ndarray).
    fs:  Sampling rate in Hz.

    Returns
    -------
    float — background MOS score.
    """
    return _run_dnsmos(wav, fs, **kwargs)['bak_mos']


# ---------------------------------------------------------------------------
# NISQA
# ---------------------------------------------------------------------------

def compute_nisqa(
    wav,
    fs: int = 16000,
    *,
    device: str = 'cpu',
    **kwargs,
) -> float:
    """Compute NISQA MOS score.

    The NISQA model is cached via ``_model_cache`` keyed by ``'nisqa'`` +
    ``device``.  The underlying architecture is loaded directly from:
        ``tf_restormer.utils.NISQA_models.NISQA_lib``
    The weights file is resolved relative to this file:
        ``../NISQA_models/nisqa_mos_only.tar``

    Parameters
    ----------
    wav:    Input waveform (1-D torch.Tensor or np.ndarray).
    fs:     Sampling rate in Hz (default 16000). NISQA internally uses the
            sample rate stored in the checkpoint (``args.ms_sr``).
    device: PyTorch device string (default ``'cpu'``).

    Returns
    -------
    float — NISQA MOS score (raw model output, no clamping).
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "NISQA requires 'torch'. "
            "Install with: pip install tf-restormer[metrics-nonintrusive]"
        )
    try:
        import librosa as lb
    except ImportError:
        raise ImportError(
            "NISQA requires 'librosa'. "
            "Install with: pip install tf-restormer[metrics-nonintrusive]"
        )

    # ------------------------------------------------------------------
    # Factory: build NISQA model from checkpoint (nisqa_mos_only.tar)
    # ------------------------------------------------------------------
    def _factory(dev: str):
        """Load NISQA weights and return (model, args) tuple."""
        from tf_restormer.utils.NISQA_models.NISQA_lib import NISQA

        _utils_dir = Path(__file__).parent.parent  # .../tf_restormer/utils/
        weights_path = str(_utils_dir / "NISQA_models" / "nisqa_mos_only.tar")

        checkpoint = torch.load(weights_path, map_location=dev)
        args = checkpoint['args']

        model = NISQA(
            ms_seg_length=args.ms_seg_length,
            ms_n_mels=args.ms_n_mels,
            cnn_model=args.cnn_model,
            cnn_c_out_1=args.cnn_c_out_1,
            cnn_c_out_2=args.cnn_c_out_2,
            cnn_c_out_3=args.cnn_c_out_3,
            cnn_kernel_size=args.cnn_kernel_size,
            cnn_dropout=args.cnn_dropout,
            cnn_pool_1=args.cnn_pool_1,
            cnn_pool_2=args.cnn_pool_2,
            cnn_pool_3=args.cnn_pool_3,
            cnn_fc_out_h=args.cnn_fc_out_h,
            td=args.td,
            td_sa_d_model=args.td_sa_d_model,
            td_sa_nhead=args.td_sa_nhead,
            td_sa_pos_enc=args.td_sa_pos_enc,
            td_sa_num_layers=args.td_sa_num_layers,
            td_sa_h=args.td_sa_h,
            td_sa_dropout=args.td_sa_dropout,
            td_lstm_h=args.td_lstm_h,
            td_lstm_num_layers=args.td_lstm_num_layers,
            td_lstm_dropout=args.td_lstm_dropout,
            td_lstm_bidirectional=args.td_lstm_bidirectional,
            td_2=args.td_2,
            td_2_sa_d_model=args.td_2_sa_d_model,
            td_2_sa_nhead=args.td_2_sa_nhead,
            td_2_sa_pos_enc=args.td_2_sa_pos_enc,
            td_2_sa_num_layers=args.td_2_sa_num_layers,
            td_2_sa_h=args.td_2_sa_h,
            td_2_sa_dropout=args.td_2_sa_dropout,
            td_2_lstm_h=args.td_2_lstm_h,
            td_2_lstm_num_layers=args.td_2_lstm_num_layers,
            td_2_lstm_dropout=args.td_2_lstm_dropout,
            td_2_lstm_bidirectional=args.td_2_lstm_bidirectional,
            pool=args.pool,
            pool_att_h=args.pool_att_h,
            pool_att_dropout=args.pool_att_dropout,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(dev)
        model.eval()
        return model, args

    model, args = _model_cache.get_or_create('nisqa', device, _factory)

    # ------------------------------------------------------------------
    # Preprocessing: waveform → mel-spectrogram segments
    # ------------------------------------------------------------------
    from tf_restormer.utils.NISQA_models.NISQA_lib import segment_specs

    # Convert to float32 numpy
    sample = _to_numpy_1d(wav)

    # Resample to the model's expected sample rate (stored in checkpoint args)
    ms_sr = int(getattr(args, 'ms_sr', 48000))
    if fs != ms_sr:
        sample = lb.resample(sample, orig_sr=fs, target_sr=ms_sr)

    # Compute mel-spectrogram (matching util_nisqa defaults)
    ms_n_fft = getattr(args, 'ms_n_fft', 1024)
    ms_hop_length = getattr(args, 'ms_hop_length', 80)
    ms_win_length = getattr(args, 'ms_win_length', 170)
    ms_n_mels = args.ms_n_mels
    ms_fmax = getattr(args, 'ms_fmax', 16000.0)

    hop_length_samples = int(ms_sr * ms_hop_length / 1000) if ms_hop_length < 1 else int(ms_sr * ms_hop_length)
    win_length_samples = int(ms_sr * ms_win_length / 1000) if ms_win_length < 1 else int(ms_sr * ms_win_length)

    mel = lb.feature.melspectrogram(
        y=sample,
        sr=ms_sr,
        n_fft=ms_n_fft,
        hop_length=hop_length_samples,
        win_length=win_length_samples,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=1.0,
        n_mels=ms_n_mels,
        fmin=0.0,
        fmax=ms_fmax,
        htk=False,
        norm='slaney',
    )
    spec = lb.amplitude_to_db(mel, ref=1.0, amin=1e-4, top_db=80.0)

    # Segment the spectrogram
    ms_seg_length = args.ms_seg_length
    ms_seg_hop_length = getattr(args, 'ms_seg_hop_length', 1)
    ms_max_segments = getattr(args, 'ms_max_segments', None)

    x_spec_seg, n_wins = segment_specs(
        file_path='<array>',
        x=spec,
        seg_length=ms_seg_length,
        seg_hop=ms_seg_hop_length,
        max_length=ms_max_segments,
    )

    # Add batch dimension and run model
    x_batch = x_spec_seg.unsqueeze(0).to(device)
    n_wins_batch = torch.tensor([n_wins]).to(device)

    with torch.inference_mode():
        mos = model(x_batch, n_wins_batch)

    return float(mos.squeeze().cpu().item())
