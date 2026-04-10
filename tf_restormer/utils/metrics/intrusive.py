"""
tf_restormer.utils.metrics.intrusive
=====================================
Intrusive (reference-required) speech quality metrics.

All functions accept both ``torch.Tensor`` and ``np.ndarray`` as input.
External dependencies are lazy-imported inside each function to avoid
import-time failures when optional packages are not installed.

Functions
---------
compute_pesq(estim, target, fs=16000, *, mode="wb", **kwargs) -> float
compute_stoi(estim, target, fs=16000, **kwargs) -> float
compute_sdr(estim, target, fs=16000, **kwargs) -> float (or np.ndarray)
compute_lsd(estim, target, fs=16000, **kwargs) -> float
compute_mcd(estim, target, fs=16000, **kwargs) -> float
compute_composite(estim, target, fs=16000, **kwargs) -> tuple[float, float, float]
    Returns (CSIG, CBAK, COVL).

MCD private helpers (merged from util_mcd.py)
---------------------------------------------
_sptk_extract(x, fs, n_fft, n_shift, mcep_dim, mcep_alpha, is_padding)
_get_best_mcep_params(fs)
_calculate_mcd(inf_audio, ref_audio, fs, n_fft, n_shift, mcep_dim, mcep_alpha)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert torch.Tensor to numpy ndarray if necessary."""
    try:
        import torch
        if torch.is_tensor(x):
            return x.cpu().detach().numpy()
    except ImportError:
        pass
    return x


# ---------------------------------------------------------------------------
# PESQ
# ---------------------------------------------------------------------------

def compute_pesq(estim, target, fs: int = 16000, *, mode: str = "wb", **kwargs) -> float:
    """Compute PESQ score.

    Parameters
    ----------
    estim:  Enhanced/estimated waveform (1-D or batch).
    target: Clean reference waveform (same shape as estim).
    fs:     Sampling rate in Hz (default 16000).
    mode:   ``'wb'`` (wideband) or ``'nb'`` (narrowband). Default ``'wb'``.

    Returns
    -------
    float — mean PESQ score across batch dimension.
    """
    try:
        import pesq as pesq_lib
    except ImportError:
        raise ImportError(
            "PESQ requires 'pesq'. "
            "Install with: pip install tf-restormer[metrics-intrusive]"
        )

    estim = _to_numpy(estim)
    target = _to_numpy(target)

    if mode == "wb":
        val = pesq_lib.pesq_batch(
            fs, target, estim, 'wb',
            on_error=pesq_lib.PesqError.RETURN_VALUES, n_processor=20,
        )
    elif mode == "nb":
        val = pesq_lib.pesq_batch(
            fs, target, estim, 'nb',
            on_error=pesq_lib.PesqError.RETURN_VALUES, n_processor=20,
        )
    else:
        # Both wb and nb — average all scores
        val_wb = pesq_lib.pesq_batch(
            fs, target, estim, 'wb',
            on_error=pesq_lib.PesqError.RETURN_VALUES, n_processor=20,
        )
        val_nb = pesq_lib.pesq_batch(
            fs, target, estim, 'nb',
            on_error=pesq_lib.PesqError.RETURN_VALUES, n_processor=20,
        )
        val = val_wb + val_nb

    return sum(val) / len(val)


# ---------------------------------------------------------------------------
# STOI
# ---------------------------------------------------------------------------

def compute_stoi(estim, target, fs: int = 16000, **kwargs) -> float:
    """Compute STOI score.

    Parameters
    ----------
    estim:  Enhanced waveform (1-D).
    target: Clean reference waveform (1-D).
    fs:     Sampling rate in Hz (default 16000).

    Returns
    -------
    float — STOI score in [0, 1].
    """
    try:
        from pystoi.stoi import stoi as _stoi
    except ImportError:
        raise ImportError(
            "STOI requires 'pystoi'. "
            "Install with: pip install tf-restormer[metrics-intrusive]"
        )

    estim = _to_numpy(estim).squeeze()
    target = _to_numpy(target).squeeze()

    return _stoi(target, estim, fs, extended=False)


# ---------------------------------------------------------------------------
# SDR
# ---------------------------------------------------------------------------

def compute_sdr(estim, target, fs: int = 16000, **kwargs):
    """Compute Signal-to-Distortion Ratio (SDR) via mir_eval BSS eval.

    Parameters
    ----------
    estim:  Enhanced waveform (1-D or [n_src, n_samples]).
    target: Clean reference waveform (same shape).
    fs:     Sampling rate in Hz (unused, kept for dispatcher signature).

    Returns
    -------
    np.ndarray — SDR values (one per source).
    """
    try:
        from mir_eval.separation import bss_eval_sources
    except ImportError:
        raise ImportError(
            "SDR requires 'mir-eval'. "
            "Install with: pip install tf-restormer[metrics-intrusive]"
        )

    estim = _to_numpy(estim)
    target = _to_numpy(target)

    SDR, _, _, _ = bss_eval_sources(target, estim)
    return SDR


# ---------------------------------------------------------------------------
# LSD
# ---------------------------------------------------------------------------

def compute_lsd(estim, target, fs: int = 16000, **kwargs) -> float:
    """Compute Log-Spectral Distance (LSD).

    Self-contained — uses PyTorch STFT only (no external optional dep).

    Parameters
    ----------
    estim:  Enhanced waveform tensor or array.
    target: Clean reference waveform tensor or array.
    fs:     Sampling rate in Hz (unused, kept for dispatcher signature).

    Returns
    -------
    float — LSD value.
    """
    import torch

    def _stft(audio, n_fft: int = 2048, hop_length: int = 512):
        hann_window = torch.hann_window(n_fft).to(audio.device)
        stft_spec = torch.stft(
            audio, n_fft, hop_length, window=hann_window, return_complex=True
        )
        stft_mag = torch.abs(stft_spec)
        return stft_mag

    if not torch.is_tensor(estim):
        estim = torch.tensor(estim, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    sp = torch.log10(_stft(estim).square().clamp(1e-8))
    st = torch.log10(_stft(target).square().clamp(1e-8))
    lsd = (sp - st).square().mean(dim=1).sqrt().mean()

    return float(lsd.cpu().detach().numpy())


# ---------------------------------------------------------------------------
# MCD — private helpers merged from util_mcd.py
# ---------------------------------------------------------------------------

def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    """Return (mcep_dim, mcep_alpha) for a given sampling rate.

    Reference:
        https://sp-nitech.github.io/sptk/latest/main/mgcep.html
    """
    table = {
        8000:  (13, 0.31),
        16000: (23, 0.42),
        22050: (34, 0.45),
        24000: (34, 0.46),
        32000: (36, 0.50),
        44100: (39, 0.53),
        48000: (39, 0.55),
    }
    if fs not in table:
        raise ValueError(
            f"No MCEP parameter preset for fs={fs}. "
            f"Supported rates: {sorted(table.keys())}"
        )
    return table[fs]


def _sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.

    Originally ``sptk_extract()`` in ``util_mcd.py``. Merged here as a private
    helper so that ``util_mcd.py`` can be safely deleted in Step 4.5.2.

    Args:
        x (ndarray): 1-D waveform array.
        fs (int):    Sampling rate.
        n_fft (int): FFT length in samples (default 512).
        n_shift (int): Hop length in samples (default 256).
        mcep_dim (int):   Mel-cepstrum order (default 25).
        mcep_alpha (float): All-pass filter coefficient (default 0.41).
        is_padding (bool): Whether to pad the end of the signal (default False).

    Returns:
        ndarray: Mel-cepstrum, shape (n_frames, mcep_dim+1).
    """
    try:
        import pysptk
    except ImportError:
        raise ImportError(
            "MCD requires 'pysptk'. "
            "Install with: pip install tf-restormer[metrics-intrusive]"
        )
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "MCD requires 'joblib'. "
            "Install with: pip install tf-restormer[metrics-intrusive]"
        )

    # Resolve None params
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # Optional end padding
    if is_padding:
        remain = (len(x) - n_fft) % n_shift
        n_pad = (n_shift - remain) % n_shift
        x = np.pad(x, (0, n_pad), mode="reflect")

    # Build frame matrix
    try:
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::n_shift]
    except AttributeError:
        # Fallback for older NumPy versions
        n_frame = (len(x) - n_fft) // n_shift + 1
        frames = np.stack(
            [x[n_shift * i: n_shift * i + n_fft] for i in range(n_frame)],
            axis=0,
        )

    win = pysptk.sptk.hamming(n_fft)

    # Parallel mcep extraction (use all cores)
    mcep_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(pysptk.mcep)(
            f * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for f in frames
    )

    return np.stack(mcep_list)


def _calculate_mcd(
    inf_audio: np.ndarray,
    ref_audio: np.ndarray,
    fs: int,
    n_fft: int = 1024,
    n_shift: int = 256,
    mcep_dim=None,
    mcep_alpha=None,
) -> float:
    """Compute Mel Cepstral Distortion between two waveforms.

    Originally ``calculate()`` in ``util_mcd.py``. Merged here as a private
    helper.

    Args:
        inf_audio: Enhanced waveform (1-D numpy array).
        ref_audio: Reference waveform (1-D numpy array).
        fs:        Sampling rate.
        n_fft:     FFT length (default 1024).
        n_shift:   Hop length (default 256).
        mcep_dim:  Mel-cepstrum order. If None, auto-selected from fs.
        mcep_alpha: All-pass filter coefficient. If None, auto-selected.

    Returns:
        float — MCD value (lower is better).
    """
    gen_mcep = _sptk_extract(
        x=inf_audio,
        fs=fs,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )
    gt_mcep = _sptk_extract(
        x=ref_audio,
        fs=fs,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )

    # Direct frame-wise MCD (no DTW alignment — same as util_mcd.py)
    diff2sum = np.sum((gen_mcep - gt_mcep) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
    return float(mcd)


# ---------------------------------------------------------------------------
# MCD — public function
# ---------------------------------------------------------------------------

def compute_mcd(estim, target, fs: int = 16000, eps: float = 1e-8, **kwargs) -> float:
    """Compute Mel Cepstral Distortion (MCD).

    Applies a global scaling factor to align energy before computing MCD,
    consistent with the logic in ``util_metric.py::MCD()``.

    Parameters
    ----------
    estim:  Enhanced waveform (1-D tensor or array).
    target: Clean reference waveform (1-D tensor or array).
    fs:     Sampling rate in Hz (default 16000).
    eps:    Numerical stability epsilon for scaling (default 1e-8).

    Returns
    -------
    float — MCD value.
    """
    estim = _to_numpy(estim).squeeze()
    target = _to_numpy(target).squeeze()

    # Energy-based scaling (from util_metric.py::MCD)
    scaling_factor = np.sum(target * estim) / (np.sum(estim ** 2) + eps)
    estim_scaled = estim * scaling_factor

    return _calculate_mcd(estim_scaled, target, fs)


# ---------------------------------------------------------------------------
# Composite metrics (CSIG, CBAK, COVL)
# Migrated from util_composite.py::compute_metrics()
# ---------------------------------------------------------------------------

def compute_composite(
    estim, target, fs: int = 16000, **kwargs
) -> Tuple[float, float, float]:
    """Compute composite speech quality metrics.

    Implements the CSIG / CBAK / COVL composite measures originally from:
        Hu & Loizou (2008), "Evaluation of Objective Quality Measures for
        Speech Enhancement", IEEE TASLP.

    Source: ``util_composite.py::compute_metrics()`` (merged here verbatim,
    with lazy imports replacing the top-level imports).

    Parameters
    ----------
    estim:  Enhanced waveform (torch.Tensor or np.ndarray, 1-D).
    target: Clean reference waveform (same shape).
    fs:     Sampling rate in Hz (default 16000).

    Returns
    -------
    tuple[float, float, float] — (CSIG, CBAK, COVL), each clipped to [1, 5].
    """
    try:
        from scipy.io import wavfile as _wavfile
        from scipy.linalg import toeplitz, norm
        from scipy.fftpack import fft
        from scipy import signal
    except ImportError:
        raise ImportError(
            "composite metrics require 'scipy'. "
            "Install with: pip install scipy"
        )
    try:
        import pesq as pesq_lib
    except ImportError:
        raise ImportError(
            "composite metrics require 'pesq'. "
            "Install with: pip install tf-restormer[metrics-intrusive]"
        )

    # Convert inputs to numpy float arrays
    try:
        import torch
        if torch.is_tensor(estim):
            estim = estim.squeeze().cpu().numpy()
        if torch.is_tensor(target):
            target = target.squeeze().cpu().numpy()
    except ImportError:
        pass

    if isinstance(estim, np.ndarray):
        estim = estim.squeeze()
    if isinstance(target, np.ndarray):
        target = target.squeeze()

    data1 = target.astype(np.float64)
    data2 = estim.astype(np.float64)
    sampling_rate = fs

    if len(data1) != len(data2):
        length = min(len(data1), len(data2))
        data1 = data1[:length] + np.spacing(1)
        data2 = data2[:length] + np.spacing(1)

    alpha = 0.95

    # --- WSS ---
    def _wss(clean_speech, processed_speech, sample_rate):
        clean_length = np.size(clean_speech)
        processed_length = np.size(processed_speech)
        if clean_length != processed_length:
            raise ValueError('Files must have same length.')

        winlength = int(np.round(30 * sample_rate / 1000))
        skiprate = int(np.floor(winlength / 4))
        max_freq = int(sample_rate / 2)
        num_crit = 25

        n_fft = int(np.power(2, np.ceil(np.log2(2 * winlength))))
        n_fftby2 = int(0.5 * n_fft)
        Kmax = 20.0
        Klocmax = 1.0

        cent_freq = np.array([
            50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000,
            540.000, 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30,
            1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71,
            2701.97, 2978.04, 3276.17, 3597.63,
        ])
        bandwidth = np.array([
            70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000,
            77.3724, 86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423,
            153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255,
            276.072, 298.126, 321.465, 346.136,
        ])
        bw_min = bandwidth[0]
        min_factor = math.exp(-30.0 / (2.0 * 2.303))

        crit_filter = np.empty((num_crit, n_fftby2))
        for i in range(num_crit):
            f0 = (cent_freq[i] / max_freq) * n_fftby2
            bw = (bandwidth[i] / max_freq) * n_fftby2
            norm_factor = np.log(bw_min) - np.log(bandwidth[i])
            j = np.arange(n_fftby2)
            crit_filter[i, :] = np.exp(
                -11 * np.square(np.divide(j - np.floor(f0), bw)) + norm_factor
            )
            cond = np.greater(crit_filter[i, :], min_factor)
            crit_filter[i, :] = np.where(cond, crit_filter[i, :], 0)

        num_frames = int(clean_length / skiprate - (winlength / skiprate))
        start = 0
        window = 0.5 * (1 - np.cos(
            2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)
        ))
        distortion = np.empty(num_frames)

        for frame_count in range(num_frames):
            clean_frame = clean_speech[start: start + winlength] / 32768
            processed_frame = processed_speech[start: start + winlength] / 32768
            clean_frame = np.multiply(clean_frame, window)
            processed_frame = np.multiply(processed_frame, window)

            clean_spec = np.square(np.abs(fft(clean_frame, n_fft)))
            processed_spec = np.square(np.abs(fft(processed_frame, n_fft)))

            clean_energy = np.matmul(crit_filter, clean_spec[0:n_fftby2])
            processed_energy = np.matmul(crit_filter, processed_spec[0:n_fftby2])

            clean_energy = 10 * np.log10(np.maximum(clean_energy, 1E-10))
            processed_energy = 10 * np.log10(np.maximum(processed_energy, 1E-10))

            clean_slope = clean_energy[1:num_crit] - clean_energy[0: num_crit - 1]
            processed_slope = processed_energy[1:num_crit] - processed_energy[0: num_crit - 1]

            clean_loc_peak = np.empty(num_crit - 1)
            processed_loc_peak = np.empty(num_crit - 1)
            for i in range(num_crit - 1):
                if clean_slope[i] > 0:
                    n = i
                    while (n < num_crit - 1) and (clean_slope[n] > 0):
                        n += 1
                    clean_loc_peak[i] = clean_energy[n - 1]
                else:
                    n = i
                    while (n >= 0) and (clean_slope[n] <= 0):
                        n -= 1
                    clean_loc_peak[i] = clean_energy[n + 1]

                if processed_slope[i] > 0:
                    n = i
                    while (n < num_crit - 1) and (processed_slope[n] > 0):
                        n += 1
                    processed_loc_peak[i] = processed_energy[n - 1]
                else:
                    n = i
                    while (n >= 0) and (processed_slope[n] <= 0):
                        n -= 1
                    processed_loc_peak[i] = processed_energy[n + 1]

            dBMax_clean = np.max(clean_energy)
            dBMax_processed = np.max(processed_energy)

            Wmax_clean = np.divide(Kmax, Kmax + dBMax_clean - clean_energy[0: num_crit - 1])
            Wlocmax_clean = np.divide(
                Klocmax, Klocmax + clean_loc_peak - clean_energy[0: num_crit - 1]
            )
            W_clean = np.multiply(Wmax_clean, Wlocmax_clean)

            Wmax_processed = np.divide(
                Kmax, Kmax + dBMax_processed - processed_energy[0: num_crit - 1]
            )
            Wlocmax_processed = np.divide(
                Klocmax, Klocmax + processed_loc_peak - processed_energy[0: num_crit - 1]
            )
            W_processed = np.multiply(Wmax_processed, Wlocmax_processed)

            W = np.divide(np.add(W_clean, W_processed), 2.0)
            slope_diff = np.subtract(clean_slope, processed_slope)[0: num_crit - 1]
            distortion[frame_count] = np.dot(W, np.square(slope_diff)) / np.sum(W)
            start += skiprate

        return distortion

    # --- LLR ---
    def _lpcoeff(speech_frame, model_order):
        winlength = np.size(speech_frame)
        R = np.empty(model_order + 1)
        E = np.empty(model_order + 1)
        for k in range(model_order + 1):
            R[k] = np.dot(speech_frame[0: winlength - k], speech_frame[k: winlength])

        a = np.ones(model_order)
        a_past = np.empty(model_order)
        rcoeff = np.empty(model_order)
        E[0] = R[0]
        for i in range(model_order):
            a_past[0: i] = a[0: i]
            sum_term = np.dot(a_past[0: i], R[i:0:-1])
            rcoeff[i] = (R[i + 1] - sum_term) / E[i]
            a[i] = rcoeff[i]
            if i == 0:
                a[0: i] = a_past[0: i] - np.multiply(a_past[i - 1:-1:-1], rcoeff[i])
            else:
                a[0: i] = a_past[0: i] - np.multiply(a_past[i - 1::-1], rcoeff[i])
            E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]

        acorr = R
        refcoeff = rcoeff
        lpparams = np.concatenate((np.array([1]), -a))
        return acorr, refcoeff, lpparams

    def _llr(clean_speech, processed_speech, sample_rate):
        clean_length = np.size(clean_speech)
        processed_length = np.size(processed_speech)
        if clean_length != processed_length:
            raise ValueError('Both Speech Files must be same length.')

        winlength = int(np.round(30 * sample_rate / 1000))
        skiprate = int(np.floor(winlength / 4))
        P = 10 if sample_rate < 10000 else 16

        num_frames = int((clean_length - winlength) / skiprate)
        start = 0
        window = 0.5 * (1 - np.cos(
            2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)
        ))
        distortion = np.empty(num_frames)

        for frame_count in range(num_frames):
            clean_frame = np.multiply(
                clean_speech[start: start + winlength], window
            )
            processed_frame = np.multiply(
                processed_speech[start: start + winlength], window
            )
            R_clean, _, A_clean = _lpcoeff(clean_frame, P)
            R_processed, _, A_processed = _lpcoeff(processed_frame, P)

            numerator = np.dot(
                np.matmul(A_processed, toeplitz(R_clean)), A_processed
            )
            denominator = np.dot(
                np.matmul(A_clean, toeplitz(R_clean)), A_clean
            )
            distortion[frame_count] = math.log(numerator / denominator)
            start += skiprate

        return distortion

    # --- SNR segmental ---
    def _snr(clean_speech, processed_speech, sample_rate):
        clean_length = len(clean_speech)
        processed_length = len(processed_speech)
        if clean_length != processed_length:
            raise ValueError('Both Speech Files must be same length.')

        overall_snr = 10 * np.log10(
            np.sum(np.square(clean_speech))
            / np.sum(np.square(clean_speech - processed_speech))
        )

        winlength = round(30 * sample_rate / 1000)
        skiprate = math.floor(winlength / 4)
        MIN_SNR = -10
        MAX_SNR = 35
        num_frames = int(clean_length / skiprate - (winlength / skiprate))
        start = 0
        window = 0.5 * (1 - np.cos(
            2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)
        ))
        segmental_snr = np.empty(num_frames)
        EPS = np.spacing(1)

        for frame_count in range(num_frames):
            clean_frame = np.multiply(
                clean_speech[start: start + winlength], window
            )
            processed_frame = np.multiply(
                processed_speech[start: start + winlength], window
            )
            signal_energy = np.sum(np.square(clean_frame))
            noise_energy = np.sum(np.square(clean_frame - processed_frame))
            segmental_snr[frame_count] = 10 * math.log10(
                signal_energy / (noise_energy + EPS) + EPS
            )
            segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
            segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)
            start += skiprate

        return overall_snr, segmental_snr

    # -----------------------------------------------------------------------
    # Main composite computation
    # -----------------------------------------------------------------------
    wss_dist_vec = _wss(data1, data2, sampling_rate)
    wss_dist_vec = np.sort(wss_dist_vec)
    wss_dist = np.mean(wss_dist_vec[0: round(np.size(wss_dist_vec) * alpha)])

    LLR_dist = _llr(data1, data2, sampling_rate)
    LLRs = np.sort(LLR_dist)
    LLR_len = round(np.size(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[0: LLR_len])

    snr_dist, segsnr_dist = _snr(data1, data2, sampling_rate)
    segSNR = np.mean(segsnr_dist)

    pesq_mos = pesq_lib.pesq(sampling_rate, data1, data2, 'wb')

    CSIG = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    CSIG = float(max(1, min(5, CSIG)))
    CBAK = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
    CBAK = float(max(1, min(5, CBAK)))
    COVL = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    COVL = float(max(1, min(5, COVL)))

    return CSIG, CBAK, COVL
