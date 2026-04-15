"""
engine_infer.py
~~~~~~~~~~~~~~~
Tensor-in / tensor-out inference engine for Speech Enhancement.

This module exposes EngineInfer, a lightweight inference-only class that
owns the STFT/iSTFT objects and the model forward pass.  All file I/O is
intentionally excluded — callers (e.g. main_infer.py) are responsible for
reading/writing audio files.
"""

from __future__ import annotations

import torch
from loguru import logger

from tf_restormer.utils import util_stft


# @logger_wraps()
class EngineInfer:
    """Inference engine — tensor-in / tensor-out API for Speech Enhancement.

    The class owns the STFT/iSTFT transforms and wraps the model forward
    pass.  It does NOT perform any file I/O; callers supply raw waveform
    tensors and receive enhanced waveform tensors in return.

    Args:
        config:  Full experiment config dict (same schema used in training).
        model:   Instantiated model (already loaded / init weights are the
                 caller's responsibility — see main_infer.py).
        device:  torch.device to run on.
        fs_src:  Output sample rate produced by the model (Hz).
        fs_in:   Input sample rate that the model expects (Hz).
    """

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        device: torch.device,
        fs_src: int = 16000,
        fs_in: int = 16000,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.fs_src = fs_src
        self.fs_in = fs_in

        self.model.eval()

        # ------------------------------------------------------------------
        # STFT / iSTFT setup
        # ------------------------------------------------------------------
        self.stft: dict = {}
        self.istft: dict = {}
        self.fs_list: list = config["fs_list"]
        for fs in self.fs_list:
            frame_len = int(config["stft"]["frame_length"] * int(fs) / 1000)
            frame_hop = int(config["stft"]["frame_shift"] * int(fs) / 1000)
            self.stft[fs] = util_stft.STFT(frame_len, frame_hop, device=self.device, normalize=True)
            self.istft[fs] = util_stft.iSTFT(frame_len, frame_hop, device=self.device, normalize=True)

        # Number of output frequency bins at the model's native output rate
        self.out_F: int = (
            int(config["stft"]["frame_length"] * int(fs_src) / 1000) // 2 + 1
        )
        # Stored separately so per-call out_F can be recomputed for a
        # different fs_out without touching self.out_F.
        self.frame_length: float = config["stft"]["frame_length"]  # ms

        # ------------------------------------------------------------------
        # Chunked-streaming defaults (can be overridden per-call)
        # ------------------------------------------------------------------
        _infer_cfg = config.get("engine", {}).get("inference", {})
        self.chunk_sec: float = _infer_cfg.get("chunk_sec", 4.0)
        self.overlap_sec: float = _infer_cfg.get("overlap_sec", 0.5)

    # ======================================================================
    # Public API
    # ======================================================================

    @torch.inference_mode()
    def infer_chunk(
        self,
        waveform_chunk: torch.Tensor,
        fs_in: int | None = None,
        fs_out: int | None = None,
    ) -> dict:
        """Process a single waveform chunk through the model.

        Args:
            waveform_chunk: Raw waveform, shape ``(1, L)`` or ``(L,)``.
                            Any device / dtype is accepted — moved and cast
                            automatically.
            fs_in:          Sample rate of the input waveform (Hz).
                            Falls back to ``self.fs_in`` when None.
            fs_out:         Desired output sample rate (Hz).  When None
                            ``self.fs_src`` is used and ``self.out_F`` is
                            reused directly (no recomputation).

        Returns:
            dict with key ``"waveform"`` → enhanced tensor, shape ``(1, L)``,
            on the same device as ``self.device``.
        """
        # ---- normalise input ----
        x = waveform_chunk.to(self.device).to(torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, L)

        # ---- resolve sample rates ----
        _fs_in = fs_in if fs_in is not None else self.fs_in
        _fs_out = fs_out if fs_out is not None else self.fs_src

        # ---- per-call out_F ----
        if fs_out is None:
            out_F = self.out_F
        else:
            out_F = int(self.frame_length * int(_fs_out) / 1000) // 2 + 1

        # ---- nearest STFT key fallback for non-standard sample rates ----
        stft_key = str(_fs_in)
        if stft_key not in self.stft:
            stft_key = min(self.fs_list, key=lambda k: abs(int(k) - _fs_in))

        # ---- nearest iSTFT key fallback (mirrors STFT fallback above) ----
        istft_key = str(_fs_out)
        if istft_key not in self.istft:
            istft_key = min(self.fs_list, key=lambda k: abs(int(k) - _fs_out))

        # ---- STFT -> model -> iSTFT pipeline ----
        X = self.stft[stft_key](x, cplx=True)                         # (1, F, T)
        model_input = torch.stack([torch.real(X), torch.imag(X)], dim=-1)  # (1, F, T, 2)
        comp = self.model(model_input, out_F=out_F)                    # (1, F, T, 2)
        out_cplx = torch.complex(comp[..., 0], comp[..., 1])           # (1, F, T)
        out_wav = self.istft[istft_key](out_cplx, cplx=True, squeeze=False)  # (1, L)

        return {"waveform": out_wav}

    @torch.inference_mode()
    def infer_session(
        self,
        waveform: torch.Tensor,
        fs_in: int | None = None,
        fs_out: int | None = None,
        mode: str = "auto",
        css_config: dict | None = None,
        show_progress: bool = False,
    ) -> dict:
        """Enhance a full utterance or recording.

        Args:
            waveform:       1-D or 2-D (1, L) waveform tensor.
            fs_in:          Input sample rate (Hz).  Defaults to ``self.fs_in``.
            fs_out:         Output sample rate (Hz).  Defaults to ``self.fs_src``.
            mode:           ``"auto"``  — choose based on waveform length;
                            ``"css"``  — always use chunked overlap-add;
                            ``"single_pass"``  — always process in one shot.
            css_config:     Optional dict to override ``chunk_sec`` /
                            ``overlap_sec`` for this call only.
            show_progress:  Show tqdm progress bar during CSS chunking.

        Returns:
            dict with key ``"waveform"`` → enhanced tensor, shape ``(1, L)``.
        """
        _fs_in = fs_in if fs_in is not None else self.fs_in
        _fs_out = fs_out if fs_out is not None else self.fs_src

        wav = self._preprocess_waveform(waveform)

        threshold_samples = int(self.chunk_sec * _fs_in)

        if mode == "auto":
            use_css = wav.shape[0] > threshold_samples
        elif mode == "css":
            use_css = True
        elif mode == "single_pass":
            use_css = False
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose from 'auto', 'css', 'single_pass'.")

        if use_css:
            enhanced = self._css_session(
                wav,
                fs_in=_fs_in,
                fs_out=_fs_out,
                css_config=css_config,
                show_progress=show_progress,
            )
        else:
            result = self._single_pass_session(wav, fs_in=_fs_in, fs_out=_fs_out)
            enhanced = result["waveform"]  # (1, L)

        return {"waveform": enhanced}

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _css_session(
        self,
        waveform: torch.Tensor,
        fs_in: int,
        fs_out: int | None = None,
        css_config: dict | None = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Chunked-streaming synthesis (CSS) with Hann-fade overlap-add.

        Args:
            waveform:   1-D tensor ``(L,)`` on any device / dtype.
            fs_in:      Input sample rate (Hz).
            fs_out:     Output sample rate (Hz).  Defaults to ``self.fs_src``.
            css_config: Optional dict with ``chunk_sec`` / ``overlap_sec``
                        overrides for this call.
            show_progress: Show tqdm progress bar.

        Returns:
            Enhanced waveform tensor, shape ``(1, L_out)`` on ``self.device``,
            where ``L_out`` is in **output sample rate (fs_out) space**.
            When ``fs_out == fs_in``, ``L_out == L_in`` (no change).
        """
        _fs_out = fs_out if fs_out is not None else self.fs_src

        # ---- resolve chunk / overlap lengths (input space, fs_in) ----
        _cfg = css_config or {}
        chunk_sec = _cfg.get("chunk_sec", self.chunk_sec)
        overlap_sec = _cfg.get("overlap_sec", self.overlap_sec)

        chunk_len = int(chunk_sec * fs_in)
        overlap_len = int(overlap_sec * fs_in)
        hop_len = chunk_len - overlap_len
        total_len = waveform.shape[0]

        # Short clip: fall back to single-pass
        if total_len <= chunk_len:
            return self.infer_chunk(
                waveform.unsqueeze(0), fs_in=fs_in, fs_out=_fs_out
            )["waveform"]  # (1, L_out)

        # ---- output-space sizing (handles fs_out != fs_in) ----
        out_ratio = _fs_out / fs_in
        out_chunk_len = int(chunk_sec * _fs_out)
        out_overlap_len = int(overlap_sec * _fs_out)
        out_total_len = int(total_len * out_ratio)

        # Accumulation buffers on CPU to avoid OOM for long recordings
        out_wav = torch.zeros(out_total_len, dtype=torch.float32)
        weight = torch.zeros(out_total_len, dtype=torch.float32)

        # Hann-based linear fade window — sized for OUTPUT chunk length
        fade = torch.ones(out_chunk_len, dtype=torch.float32)
        if out_overlap_len > 0:
            fade[:out_overlap_len] = torch.linspace(0.0, 1.0, out_overlap_len)
            fade[-out_overlap_len:] = torch.linspace(1.0, 0.0, out_overlap_len)

        # ---- optional progress bar ----
        positions = list(range(0, total_len, hop_len))
        if show_progress:
            try:
                from tqdm import tqdm as _tqdm
                positions = _tqdm(positions, unit="chunk", dynamic_ncols=True, colour="CYAN")
            except ImportError:
                logger.warning("tqdm not installed — show_progress ignored.")

        # ---- chunked processing loop ----
        # Input chunking stays in fs_in space; output accumulation uses fs_out space.
        for start in positions:
            end = min(start + chunk_len, total_len)
            chunk = waveform[start:end]

            pad_len = chunk_len - chunk.shape[0]
            if pad_len > 0:
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            enhanced_chunk = self.infer_chunk(
                chunk.unsqueeze(0), fs_in=fs_in, fs_out=_fs_out
            )["waveform"].squeeze(0).cpu()  # (out_chunk_len,)

            # Map input position to output position via sample-rate ratio
            out_start = int(start * out_ratio)
            # Clamp to prevent overflow on the last (zero-padded) chunk
            out_actual_len = min(enhanced_chunk.shape[0], out_total_len - out_start)
            out_end = out_start + out_actual_len

            w = fade[:out_actual_len]

            out_wav[out_start:out_end] += enhanced_chunk[:out_actual_len] * w
            weight[out_start:out_end] += w

        # Normalise by accumulated weight (avoid div-by-zero at edges)
        weight = weight.clamp(min=1e-8)
        out_wav = out_wav / weight

        return out_wav.unsqueeze(0).to(self.device)  # (1, L_out)

    def _single_pass_session(
        self,
        waveform: torch.Tensor,
        fs_in: int | None = None,
        fs_out: int | None = None,
    ) -> dict:
        """Single forward pass for short audio.

        Args:
            waveform: 1-D tensor ``(L,)``.
            fs_in:    Input sample rate (Hz).  Defaults to ``self.fs_in``.
            fs_out:   Output sample rate (Hz).  Defaults to ``self.fs_src``.

        Returns:
            dict with key ``"waveform"`` → shape ``(1, L)``.
        """
        return self.infer_chunk(
            waveform.unsqueeze(0),
            fs_in=fs_in,
            fs_out=fs_out,
        )

    def _preprocess_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform shape and transfer to device.

        Ensures consistent (L,) shape and float32 dtype on self.device.
        SE variant: identity-like — only shape/dtype/device normalization.

        Args:
            waveform: 1-D (L,) or 2-D (1, L) waveform tensor.

        Returns:
            Tensor of shape (L,) on self.device with float32 dtype.
        """
        wav = waveform.to(self.device).to(torch.float32)
        if wav.dim() == 2:
            wav = wav.squeeze(0)  # (1, L) -> (L,)
        return wav
