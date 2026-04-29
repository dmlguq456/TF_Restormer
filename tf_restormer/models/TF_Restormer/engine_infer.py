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
        _infer_cfg = config.get("inference", {})
        self.chunk_sec: float = _infer_cfg.get("chunk_sec", 4.0)
        self.overlap_sec: float = _infer_cfg.get("overlap_sec", 0.5)
        # STFT-frame chunk config (optional — resolved lazily by _resolve_chunk_config)
        self._stft_chunk_config: dict | None = None
        if "N_h" in _infer_cfg and "N_c" in _infer_cfg and "N_f" in _infer_cfg:
            self._stft_chunk_config = {
                "N_h": int(_infer_cfg["N_h"]),
                "N_c": max(1, int(_infer_cfg["N_c"])),
                "N_f": int(_infer_cfg["N_f"]),
            }

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

        Thin wrapper around ``infer_stft_chunk``: performs STFT on the input
        waveform, delegates the model forward to ``infer_stft_chunk``, then
        applies iSTFT to produce the output waveform.

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

        # ---- STFT key lookup (nearest fallback for non-standard rates) ----
        stft_key = self._resolve_stft_key(_fs_in)
        istft_key = self._resolve_stft_key(_fs_out)

        # ---- STFT -> infer_stft_chunk -> iSTFT ----
        X = self.stft[stft_key](x, cplx=True)                              # (1, F, T)
        out_cplx = self.infer_stft_chunk(X, fs_out=_fs_out)                # (1, F_out, T)
        out_wav = self.istft[istft_key](out_cplx, cplx=True, squeeze=False)  # (1, L)

        return {"waveform": out_wav}

    @torch.inference_mode()
    def infer_stft_chunk(
        self,
        stft_chunk: torch.Tensor,
        fs_out: int | None = None,
    ) -> torch.Tensor:
        """Run model forward on a pre-computed STFT chunk (no STFT/iSTFT).

        This is the core model-forward operation extracted from ``infer_chunk``.
        Callers that already hold a complex STFT tensor (e.g. ``_css_session``
        and ``InferenceSession``) should call this directly to avoid redundant
        STFT/iSTFT round-trips.

        Args:
            stft_chunk: Complex STFT tensor ``(1, F, T_chunk)`` on
                        ``self.device``.  Any dtype is accepted — cast to
                        complex64 automatically if needed.
            fs_out:     Output sample rate (Hz) used to compute the number of
                        output frequency bins (``out_F``).  Defaults to
                        ``self.fs_src``.

        Returns:
            Complex STFT tensor ``(1, F_out, T_chunk)`` on ``self.device``.
        """
        _fs_out = fs_out if fs_out is not None else self.fs_src

        # ---- per-call out_F ----
        if fs_out is None:
            out_F = self.out_F
        else:
            out_F = int(self.frame_length * int(_fs_out) / 1000) // 2 + 1

        # ---- model forward: complex → stacked → model → complex ----
        model_input = torch.stack(
            [torch.real(stft_chunk), torch.imag(stft_chunk)], dim=-1
        )  # (1, F, T, 2)
        comp = self.model(model_input, out_F=out_F)  # (1, F_out, T, 2)
        return torch.complex(comp[..., 0], comp[..., 1])  # (1, F_out, T)

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
        """Chunked-streaming synthesis (CSS) using STFT-domain context-aware chunking.

        Performs a single global STFT on the full waveform, then iterates over
        overlapping STFT-frame chunks (N_h|N_c|N_f pattern) using
        ``stft_chunk_generator`` and ``SEChunkStitcher``.  A single global iSTFT
        is applied to the stitched output, avoiding per-chunk STFT/iSTFT overhead
        and boundary artifacts.

        Args:
            waveform:      1-D tensor ``(L,)`` on any device / dtype.
            fs_in:         Input sample rate (Hz).
            fs_out:        Output sample rate (Hz).  Defaults to ``self.fs_src``.
            css_config:    Optional dict to override chunk parameters.  Accepts
                           ``N_h``/``N_c``/``N_f`` (frames) or
                           ``chunk_sec``/``overlap_sec`` (seconds, auto-converted).
            show_progress: Show tqdm progress bar during the chunk loop.

        Returns:
            Enhanced waveform tensor, shape ``(1, L_out)`` on ``self.device``,
            where ``L_out`` is in **output sample rate (fs_out) space**.
            When ``fs_out == fs_in``, ``L_out == L_in`` (no change in length).
        """
        from tf_restormer.utils.util_engine import stft_chunk_generator, SEChunkStitcher

        _fs_out = fs_out if fs_out is not None else self.fs_src

        # ---- resolve STFT keys ----
        stft_key = self._resolve_stft_key(fs_in)
        istft_key = self._resolve_stft_key(_fs_out)

        # ---- resolve N_h / N_c / N_f ----
        chunk_cfg = self._resolve_chunk_config(css_config, fs_in)
        N_h = chunk_cfg["N_h"]
        N_c = chunk_cfg["N_c"]
        N_f = chunk_cfg["N_f"]
        chunk_len_frames = N_h + N_c + N_f  # total frames per chunk

        # ---- global STFT (single pass) ----
        wav = waveform.to(self.device).to(torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1, L)
        stft_in = self.stft[stft_key](wav, cplx=True)  # (1, F_in, T)
        total_frames = stft_in.shape[2]

        # ---- short-clip fallback: delegate to single-pass ----
        if total_frames <= chunk_len_frames:
            return self._single_pass_session(
                waveform, fs_in=fs_in, fs_out=_fs_out
            )["waveform"]  # (1, L_out)

        # ---- expected output length in waveform samples ----
        out_total_len = int(waveform.shape[0] * _fs_out / fs_in)

        # ---- build chunk list via generator ----
        gen_out = stft_chunk_generator(chunk_cfg, stft_in)
        chunk_list, _shift, _N_h, _N_f, dummy_frame_len, stft_pad = gen_out

        # ---- stitcher for STFT-domain accumulation ----
        stitcher = SEChunkStitcher(chunk_cfg, device=self.device, use_blending=True)

        # ---- optional progress bar ----
        indexed_chunks: list | object = list(enumerate(chunk_list))
        if show_progress:
            try:
                from tqdm import tqdm as _tqdm
                indexed_chunks = _tqdm(
                    indexed_chunks, unit="chunk", dynamic_ncols=True, colour="CYAN"
                )
            except ImportError:
                logger.warning("tqdm not installed — show_progress ignored.")

        # ---- STFT-domain chunk loop ----
        # SEChunkStitcher slices on dim=-1 (last dim), so pass complex (1, F, T)
        # tensors directly — time is the last dimension for complex tensors.
        for chunk_idx, (begin, end) in indexed_chunks:
            stft_chunk = stft_pad[..., begin:end]  # (1, F_in, chunk_len_frames)
            out_cplx = self.infer_stft_chunk(stft_chunk, fs_out=_fs_out)  # (1, F_out, T_chunk)
            stitcher.add_chunk(out_cplx, chunk_idx, accumulate=True)

        # ---- finalize stitcher → trim dummy frames → global iSTFT ----
        result_cplx = stitcher.finalize(dummy_frame_len)  # (1, F_out, T_out) complex
        out_wav = self.istft[istft_key](result_cplx, cplx=True, squeeze=False)  # (1, L_out_raw)

        # ---- trim to expected output length (compensate for STFT boundary padding) ----
        out_wav = out_wav[:, :out_total_len]

        return out_wav  # (1, L_out)

    def _resolve_stft_key(self, fs: int) -> str:
        """Return the nearest key in ``self.stft``/``self.istft`` for *fs*.

        If *fs* exactly matches an element of ``self.fs_list``, that element
        is returned as-is.  Otherwise, the element with the smallest absolute
        difference is returned.

        Args:
            fs: Sample rate (Hz) to look up.

        Returns:
            A key from ``self.fs_list`` (str) suitable for indexing
            ``self.stft`` or ``self.istft``.
        """
        key = str(fs)
        if key in self.stft:
            return key
        return min(self.fs_list, key=lambda k: abs(int(k) - fs))

    def _resolve_chunk_config(
        self, css_config: dict | None, fs_in: int
    ) -> dict:
        """Resolve STFT-frame chunk parameters (N_h, N_c, N_f).

        Resolution priority (highest to lowest):
        1. Explicit ``N_h``/``N_c``/``N_f`` in *css_config*.
        2. ``chunk_sec``/``overlap_sec`` in *css_config* (converted to frames).
        3. ``self._stft_chunk_config`` (N_h/N_c/N_f from config YAML).
        4. Instance defaults (``self.chunk_sec`` / ``self.overlap_sec``),
           converted to STFT frames.

        Conversion formula::

            total_frames  = int(chunk_sec  * fs_in / frame_hop)
            overlap_frames = int(overlap_sec * fs_in / frame_hop)
            N_h = N_f = overlap_frames
            N_c = total_frames - 2 * overlap_frames

        Args:
            css_config: Optional per-call override dict.
            fs_in:      Input sample rate (Hz) used for frame conversion.

        Returns:
            Dict with integer keys ``N_h``, ``N_c``, ``N_f``.
        """
        _cfg = css_config or {}

        # Priority 1: explicit frame counts in css_config
        if "N_h" in _cfg and "N_c" in _cfg and "N_f" in _cfg:
            return {
                "N_h": int(_cfg["N_h"]),
                "N_c": max(1, int(_cfg["N_c"])),
                "N_f": int(_cfg["N_f"]),
            }

        # Priority 2: sec-based values in css_config
        if "chunk_sec" in _cfg or "overlap_sec" in _cfg:
            chunk_sec = float(_cfg.get("chunk_sec", self.chunk_sec))
            overlap_sec = float(_cfg.get("overlap_sec", self.overlap_sec))
            stft_key = self._resolve_stft_key(fs_in)
            frame_hop = self.stft[stft_key].stride
            total_frames = int(chunk_sec * fs_in / frame_hop)
            overlap_frames = int(overlap_sec * fs_in / frame_hop)
            N_c = max(1, total_frames - 2 * overlap_frames)
            return {"N_h": overlap_frames, "N_c": N_c, "N_f": overlap_frames}

        # Priority 3: pre-parsed STFT chunk config from YAML
        if self._stft_chunk_config is not None:
            return dict(self._stft_chunk_config)

        # Priority 4: instance defaults
        stft_key = self._resolve_stft_key(fs_in)
        frame_hop = self.stft[stft_key].stride
        total_frames = int(self.chunk_sec * fs_in / frame_hop)
        overlap_frames = int(self.overlap_sec * fs_in / frame_hop)
        N_c = max(1, total_frames - 2 * overlap_frames)
        return {"N_h": overlap_frames, "N_c": N_c, "N_f": overlap_frames}

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
