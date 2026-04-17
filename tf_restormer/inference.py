"""Public inference API for TF_Restormer speech enhancement.

Waveform-level API::

    from tf_restormer import SEInference

    model = SEInference.from_pretrained(
        config="baseline.yaml",
        checkpoint_path="path/to/checkpoints/",
        device="cuda",
    )

    # Process a waveform tensor
    import torch
    waveform = torch.randn(1, 16000)  # (1, L) at 16 kHz input
    result = model.process_waveform(waveform)
    # result["waveform"] -> (1, L_out) at 48 kHz output

    # Process a wav file and optionally save
    result = model.process_file("noisy.wav", output_path="enhanced.wav")

STFT-level API (for advanced users managing STFT themselves)::

    # Input: complex STFT tensor (1, F, T) or stacked (1, F, T, 2)
    result = model.process_stft(stft_complex)
    # result["stft_out"] -> (1, F, T) complex tensor
    # result["waveform"] -> (1, L) float tensor

Streaming / session API::

    session = model.create_session(streaming=True)
    for chunk in microphone_stream():
        results = session.feed_waveform(chunk)
        for r in results:
            play(r["waveform"])
    _, tail = session.flush()
    if tail is not None:
        play(tail["waveform"])
"""
from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from tf_restormer import _config
from tf_restormer.models.TF_Restormer.model import Model
from tf_restormer.models.TF_Restormer.engine_infer import EngineInfer
from tf_restormer.utils.util_engine import (
    _fix_compiled_state_dict,
    _find_latest_checkpoint,
)

# ---------------------------------------------------------------------------
# Sentinel for compiled-model state_dict prefix stripping
# ---------------------------------------------------------------------------
_MODEL_TOPLEVEL_PREFIXES = ("input_embed", "encoder", "up", "decoder", "estimator")

# ---------------------------------------------------------------------------
# Default checkpoint home (mirrors export.py _DEFAULT_CKPT_HOME)
# ---------------------------------------------------------------------------
_DEFAULT_CKPT_HOME = Path(__file__).resolve().parent / "checkpoints"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_profiling_keys(state_dict: dict) -> dict:
    """Remove ``total_ops`` / ``total_params`` keys injected by ptflops/thop.

    Raw training checkpoints contain these float64 zero-tensors for every
    submodule.  They account for roughly half the key count and are not
    used during inference.

    Args:
        state_dict: State dict possibly containing profiling keys.

    Returns:
        Cleaned state dict with profiling entries removed.
    """
    cleaned = {
        k: v for k, v in state_dict.items()
        if "total_ops" not in k and "total_params" not in k
    }
    n_removed = len(state_dict) - len(cleaned)
    if n_removed > 0:
        logger.debug(f"Stripped {n_removed} profiling keys (total_ops / total_params).")
    return cleaned


# ---------------------------------------------------------------------------
# _BaseInference
# ---------------------------------------------------------------------------

class _BaseInference:
    """Variant-agnostic base class for inference frontends.

    Subclasses must define:
    - ``_VARIANT`` (str): dot-path to the model package
      (e.g. ``"tf_restormer.models.TF_Restormer"``).
    - ``_from_pretrained_impl()`` (classmethod): variant-specific factory that
      builds the model and returns an initialised instance.

    Instance creation uses ``cls.__new__(cls)`` + direct attribute assignment
    inside ``_from_pretrained_impl()``, so ``_BaseInference`` deliberately does
    **not** define ``__init__``.
    """

    _VARIANT: str  # set by subclass

    # Declare ``engine`` so that the ``device`` property below can reference
    # ``self.engine`` without type-checker errors.  The actual value is
    # assigned by ``_from_pretrained_impl()`` in the subclass.
    engine: "EngineInfer"

    # ------------------------------------------------------------------
    # HF Hub helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_hf_repo_id(path_or_id: str | None) -> bool:
        """Return True if *path_or_id* looks like a HF Hub repo ID (owner/name).

        A valid repo ID:
        - Has exactly one ``/`` separator (two parts).
        - Does NOT point to an existing local path.
        - Does NOT end with a file extension (``.pt``, ``.pth``, ``.yaml``).

        Args:
            path_or_id: Candidate string to test.

        Returns:
            ``True`` if the string matches the HF Hub repo ID pattern.
        """
        if path_or_id is None:
            return False
        s = str(path_or_id)
        if Path(s).exists():
            return False
        parts = s.split("/")
        if len(parts) != 2:
            return False
        return not any(p.endswith((".pt", ".pth", ".yaml", ".pkl")) for p in parts)

    @classmethod
    def _download_from_hub(
        cls,
        repo_id: str,
        config: str | Path | dict | None,
    ) -> tuple[Path, str | Path | dict | None]:
        """Download ``model.pt`` (and optionally ``config.yaml``) from HF Hub.

        Args:
            repo_id: HF Hub repo ID, e.g. ``"shinuh/tf-restormer-baseline"``.
            config:  Current config argument.  If ``None`` or not a local file,
                     ``config.yaml`` is also downloaded from the repo.

        Returns:
            ``(checkpoint_path, config)`` — both pointing to local cached files.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HF Hub downloads. "
                "Install with: pip install tf-restormer[hub]  "
                "or: uv sync --extra hub"
            ) from None

        ckpt_path = Path(hf_hub_download(repo_id=repo_id, filename="model.pt"))
        logger.info(f"Downloaded checkpoint from HF Hub: {repo_id}/model.pt -> {ckpt_path}")

        # Download config from repo if user did not provide a config that already
        # exists as a local file.  Extension alone is not a reliable indicator —
        # a user may pass "baseline.yaml" intending the Hub copy.
        need_config = config is None or (
            isinstance(config, str)
            and not Path(config).is_file()
        )
        if need_config:
            config = str(Path(hf_hub_download(repo_id=repo_id, filename="config.yaml")))
            logger.info(f"Downloaded config from HF Hub: {repo_id}/config.yaml -> {config}")

        return ckpt_path, config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        config: str | Path | dict = "baseline.yaml",
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        fs_src: int | None = None,
        fs_in: int | None = None,
        **kwargs,
    ) -> "_BaseInference":
        """Create an inference instance from a config and checkpoint.

        Args:
            config: Config name (e.g. ``"baseline"`` or ``"baseline.yaml"``),
                absolute path to a YAML file, or a pre-loaded config dict.
            checkpoint_path: One of:
                - Local ``.pt`` / ``.pth`` file path
                - Local directory (latest checkpoint is auto-selected)
                - HF Hub repo ID (e.g. ``"shinuh/tf-restormer-baseline"``)
                - ``None`` — looks in ``tf_restormer/checkpoints/{config_stem}/model.pt``
            device: PyTorch device string (``"cuda"``, ``"cpu"``, etc.).
            fs_src: Output sample rate the model produces (Hz).  Overrides
                the value read from the training config.  Most users should
                leave this as ``None``; the correct value (e.g. 48000) is
                read from ``config["dataset"]``.
            fs_in:  Input sample rate the model expects (Hz).  Overrides the
                training config value (e.g. 16000).
            **kwargs: Reserved for future use.

        Returns:
            Initialized instance ready for inference.

        Raises:
            FileNotFoundError: If the checkpoint cannot be located.
            ValueError: If the checkpoint format is unrecognized.
        """
        logger.disable("tf_restormer")
        try:
            return cls._from_pretrained_impl(
                config, checkpoint_path, device, fs_src, fs_in, **kwargs
            )
        finally:
            logger.enable("tf_restormer")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """PyTorch device the model runs on."""
        return self.engine.device


# ---------------------------------------------------------------------------
# InferenceSession
# ---------------------------------------------------------------------------

class InferenceSession:
    """Stateful session for chunk-by-chunk waveform processing.

    Waveform-level API:
        ``session.feed_waveform(waveform) -> list[dict]``  — returns results
        for each completed chunk.
        ``session.flush(remaining=None) -> (list[dict], dict | None)``  — drain
        remaining buffer and return the final tail chunk.

    In **batch** (non-streaming) mode, call ``finalize()`` after feeding all
    data to get the complete overlap-add reconstructed waveform.

    In **streaming** mode, consume the list returned by each
    ``feed_waveform()`` call immediately.

    Obtain a session via :meth:`SEInference.create_session`.
    """

    def __init__(
        self,
        parent: "SEInference",
        streaming: bool = False,
        fs_in: int | None = None,
        fs_out: int | None = None,
        css_config: dict | None = None,
    ) -> None:
        self._parent = parent
        self._streaming = streaming
        self._fs_in = fs_in if fs_in is not None else parent._fs_in
        self._fs_out = fs_out if fs_out is not None else parent._fs_src

        engine = parent.engine
        # Read chunk/overlap from EngineInfer (already applied fallback defaults).
        # css_config overrides are applied locally — engine state is never mutated.
        chunk_sec: float = engine.chunk_sec
        overlap_sec: float = engine.overlap_sec
        if css_config:
            chunk_sec = css_config.get("chunk_sec", chunk_sec)
            overlap_sec = css_config.get("overlap_sec", overlap_sec)

        self._chunk_len = int(chunk_sec * self._fs_in)
        self._overlap_len = int(overlap_sec * self._fs_in)
        self._hop_len = self._chunk_len - self._overlap_len

        # Output-space sizing (handles fs_out != fs_in)
        out_ratio = self._fs_out / self._fs_in
        self._out_chunk_len = int(chunk_sec * self._fs_out)
        self._out_overlap_len = int(overlap_sec * self._fs_out)
        self._out_ratio = out_ratio

        # Hann-based linear fade window sized for OUTPUT chunk length
        self._fade = torch.ones(self._out_chunk_len, dtype=torch.float32)
        if self._out_overlap_len > 0:
            self._fade[:self._out_overlap_len] = torch.linspace(0.0, 1.0, self._out_overlap_len)
            self._fade[-self._out_overlap_len:] = torch.linspace(1.0, 0.0, self._out_overlap_len)

        # Internal state
        self._wav_buffer: torch.Tensor | None = None
        self._total_input_samples: int = 0

        # Batch-mode accumulation buffers (allocated lazily on first chunk)
        self._out_buffer: torch.Tensor | None = None
        self._weight_buffer: torch.Tensor | None = None
        self._out_pos: int = 0  # current write position in output space

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def feed_waveform(
        self,
        waveform: torch.Tensor,
        fs_in: int | None = None,
        fs_out: int | None = None,
    ) -> list[dict]:
        """Buffer raw waveform samples and process complete chunks.

        Args:
            waveform: PCM float tensor of shape ``(L,)`` or ``(1, L)``.
            fs_in:    Sample rate of *waveform*.  Defaults to the session's
                      ``fs_in`` (set at :meth:`create_session` time).
            fs_out:   Desired output sample rate.  Defaults to the session's
                      ``fs_out``.

        Returns:
            list[dict]: One dict per complete chunk processed.  Each dict has
            key ``"waveform"`` → ``(1, L_out)`` enhanced tensor.
        """
        _fs_in = fs_in if fs_in is not None else self._fs_in
        _fs_out = fs_out if fs_out is not None else self._fs_out

        wav = waveform.to(self._parent.device).to(torch.float32)
        if wav.dim() == 2:
            wav = wav.squeeze(0)  # (1, L) -> (L,)

        self._total_input_samples += wav.shape[0]

        if self._wav_buffer is None:
            self._wav_buffer = wav
        else:
            self._wav_buffer = torch.cat([self._wav_buffer, wav], dim=0)

        results = []
        while self._wav_buffer.shape[0] >= self._chunk_len:
            chunk = self._wav_buffer[: self._chunk_len]
            enhanced = self._parent.engine.infer_chunk(
                chunk.unsqueeze(0), fs_in=_fs_in, fs_out=_fs_out
            )  # {"waveform": (1, L_out)}

            if self._streaming:
                results.append(enhanced)
            else:
                self._accumulate(enhanced["waveform"].squeeze(0).cpu())
                # In batch mode, return body only (overlap region excluded)
                body_start = self._out_overlap_len
                body = enhanced["waveform"][:, body_start:]
                results.append({"waveform": body})

            self._wav_buffer = self._wav_buffer[self._hop_len :]

        return results

    @torch.inference_mode()
    def flush(
        self,
        remaining_waveform: torch.Tensor | None = None,
        fs_in: int | None = None,
        fs_out: int | None = None,
    ) -> tuple[list[dict], dict | None]:
        """Process remaining buffered samples and end the stream.

        Args:
            remaining_waveform: Optional final samples to append before
                flushing.  Shape ``(L,)`` or ``(1, L)``.
            fs_in:  Sample rate of *remaining_waveform*.
            fs_out: Desired output sample rate.

        Returns:
            ``(drained, tail)`` — drained is a list of result dicts from any
            complete chunks; tail is the result for the final partial chunk,
            or ``None`` when nothing remains.
        """
        drained = []
        if remaining_waveform is not None:
            drained = self.feed_waveform(remaining_waveform, fs_in=fs_in, fs_out=fs_out)

        _fs_in = fs_in if fs_in is not None else self._fs_in
        _fs_out = fs_out if fs_out is not None else self._fs_out

        if self._wav_buffer is None or self._wav_buffer.shape[0] == 0:
            return drained, None

        tail_wav = self._wav_buffer
        self._wav_buffer = None

        result = self._parent.engine.infer_chunk(
            tail_wav.unsqueeze(0), fs_in=_fs_in, fs_out=_fs_out
        )
        return drained, result

    def finalize(self) -> dict:
        """Concatenate overlap-add buffer and return complete enhanced waveform.

        Only valid in batch (non-streaming) mode.

        Returns:
            dict with key ``"waveform"`` → ``(1, L)`` enhanced tensor.

        Raises:
            RuntimeError: If called in streaming mode.
        """
        if self._streaming:
            raise RuntimeError(
                "finalize() is not available in streaming mode. "
                "Use flush() to end the stream and collect the final chunk."
            )
        if self._out_buffer is None:
            return {"waveform": torch.zeros(1, 0)}

        weight = self._weight_buffer.clamp(min=1e-8)
        out_wav = self._out_buffer / weight

        # Trim to expected output length
        expected_out = int(self._total_input_samples * self._out_ratio)
        out_wav = out_wav[:expected_out]

        return {"waveform": out_wav.unsqueeze(0).to(self._parent.device)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_buffers(self, required_len: int) -> None:
        """Lazily allocate or extend the overlap-add accumulation buffers."""
        if self._out_buffer is None:
            buf_len = max(required_len, self._out_chunk_len * 4)
            self._out_buffer = torch.zeros(buf_len, dtype=torch.float32)
            self._weight_buffer = torch.zeros(buf_len, dtype=torch.float32)
        elif self._out_pos + self._out_chunk_len > self._out_buffer.shape[0]:
            # Extend buffers
            extra = self._out_chunk_len * 4
            self._out_buffer = torch.cat(
                [self._out_buffer, torch.zeros(extra, dtype=torch.float32)]
            )
            self._weight_buffer = torch.cat(
                [self._weight_buffer, torch.zeros(extra, dtype=torch.float32)]
            )

    def _accumulate(self, chunk: torch.Tensor) -> None:
        """Overlap-add *chunk* (1-D CPU tensor) into the accumulation buffer."""
        out_actual_len = min(chunk.shape[0], self._out_chunk_len)
        end_pos = self._out_pos + out_actual_len
        self._ensure_buffers(end_pos)

        w = self._fade[:out_actual_len]
        self._out_buffer[self._out_pos : end_pos] += chunk[:out_actual_len] * w
        self._weight_buffer[self._out_pos : end_pos] += w

        # Advance write position by hop in output space
        out_hop = int(self._hop_len * self._out_ratio)
        self._out_pos += out_hop


# ---------------------------------------------------------------------------
# SEInference
# ---------------------------------------------------------------------------

class SEInference(_BaseInference):
    """Public inference API for TF_Restormer speech enhancement.

    Waveform-level API::

        model = SEInference.from_pretrained(
            config="baseline.yaml",
            checkpoint_path="path/to/checkpoints/",
            device="cuda",
        )
        result = model.process_waveform(waveform)  # {"waveform": Tensor(1, L)}

    STFT-level API (direct model forward)::

        result = model.process_stft(stft_complex)
        # result["stft_out"] -> (1, F, T) complex tensor
        # result["waveform"] -> (1, L) float tensor

    Streaming session API::

        session = model.create_session(streaming=True)
        for chunk in audio_stream():
            results = session.feed_waveform(chunk)
        _, tail = session.flush()

    Note:
        ``fs_src`` in :class:`EngineInfer` semantics is the **output** sample
        rate the model produces (e.g., 48000 Hz for super-resolution
        enhancement).  It is NOT the rate of the source audio.  This naming
        follows the convention established in ``engine_infer.py:L48``.
    """

    _VARIANT = "tf_restormer.models.TF_Restormer"

    # ------------------------------------------------------------------
    # Construction  (variant-specific factory — called by _BaseInference.from_pretrained)
    # ------------------------------------------------------------------

    @classmethod
    def _from_pretrained_impl(
        cls,
        config,
        checkpoint_path,
        device,
        fs_src,
        fs_in,
        **kwargs,
    ) -> "SEInference":
        # ── 0. HF Hub detection ──────────────────────────────────────────
        if checkpoint_path is not None and cls._is_hf_repo_id(str(checkpoint_path)):
            checkpoint_path, config = cls._download_from_hub(
                str(checkpoint_path), config
            )

        # ── 1. Load config ────────────────────────────────────────────────
        if isinstance(config, (str, Path)):
            config_str = str(config)
            # Normalize: "baseline" -> "baseline.yaml"
            if not config_str.endswith(".yaml"):
                config_str = config_str + ".yaml"

            if Path(config_str).is_absolute() and Path(config_str).is_file():
                # Absolute path to a YAML file
                yaml_dict = _config.load_config("TF_Restormer", config_str)
            else:
                # Config name — resolve via package resources
                yaml_dict = _config.load_config("TF_Restormer", config_str)

            cfg = yaml_dict["config"]
            config_stem = Path(config_str).stem  # "baseline"
        elif isinstance(config, dict):
            # Accept either full yaml_dict or inner config dict
            if "config" in config:
                cfg = config["config"]
            else:
                cfg = config
            config_stem = None
        else:
            raise TypeError(
                f"config must be a file path (str/Path) or a dict, got {type(config)}"
            )

        # ── 2. Resolve fs_src / fs_in from training config ───────────────
        _fs_src, _fs_in = cls._resolve_sample_rates(cfg, fs_src, fs_in)

        # ── 3. Resolve checkpoint path ────────────────────────────────────
        torch_device = torch.device(device)

        if checkpoint_path is None:
            # Default: tf_restormer/checkpoints/{config_stem}/model.pt
            if config_stem is not None:
                default_pt = _DEFAULT_CKPT_HOME / config_stem / "model.pt"
            else:
                default_pt = None

            if default_pt is not None and default_pt.exists():
                ckpt_file = default_pt
            else:
                raise FileNotFoundError(
                    f"No checkpoint found at default path: {default_pt}\n"
                    f"Run export_checkpoint('{config_stem}.yaml') to export a checkpoint, "
                    f"or pass checkpoint_path explicitly."
                )
        else:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint path does not exist: {checkpoint_path}"
                )

            if checkpoint_path.is_dir():
                # Look for exported model.pt first (export_checkpoint always uses this name)
                exported_pt = checkpoint_path / "model.pt"
                if exported_pt.exists():
                    ckpt_file = exported_pt
                else:
                    result = _find_latest_checkpoint(checkpoint_path)
                    if result is None:
                        raise FileNotFoundError(
                            f"No checkpoint found in: {checkpoint_path}\n"
                            f"Expected epoch.NNNN.pth or model.pt files."
                        )
                    ckpt_file = Path(result[0])
            else:
                ckpt_file = checkpoint_path

        # ── 4. Load checkpoint ────────────────────────────────────────────
        logger.info(f"Loading checkpoint: {ckpt_file}")
        try:
            ckpt = torch.load(str(ckpt_file), map_location=torch_device, weights_only=True)
        except Exception as e:
            if "weights_only" not in str(e).lower():
                raise
            logger.warning(f"weights_only load failed, retrying without: {e}")
            ckpt = torch.load(str(ckpt_file), map_location=torch_device, weights_only=False)

        # ── 5. Detect checkpoint format and extract state_dict ────────────
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            if "optimizer_state_dict" in ckpt:
                logger.debug("Detected raw training checkpoint format (model + optimizer).")
            else:
                logger.debug("Detected exported checkpoint format (model_state_dict only).")
        elif any(k.startswith(_MODEL_TOPLEVEL_PREFIXES) for k in ckpt):
            # Bare state_dict (torch.save(model.state_dict(), path))
            state_dict = ckpt
            logger.info("Detected bare state_dict format (no 'model_state_dict' wrapper).")
        else:
            raise ValueError(
                f"Unrecognized checkpoint format. Keys: {list(ckpt.keys())[:10]}. "
                f"Expected either 'model_state_dict' key (training/exported checkpoint) "
                f"or top-level keys starting with {_MODEL_TOPLEVEL_PREFIXES} (bare state_dict)."
            )

        # Strip profiling keys and compiled prefixes
        state_dict = _strip_profiling_keys(state_dict)
        state_dict = _fix_compiled_state_dict(state_dict)

        # ── 6. Build model and load weights ───────────────────────────────
        model = Model(**cfg["model"])
        model.load_state_dict(state_dict, strict=True)
        model.to(torch_device).eval()
        logger.info(f"Model loaded successfully (device={torch_device}).")

        # ── 7. Build EngineInfer ──────────────────────────────────────────
        engine = EngineInfer(
            config=cfg,
            model=model,
            device=torch_device,
            fs_src=_fs_src,
            fs_in=_fs_in,
        )

        # ── 8. Construct instance ─────────────────────────────────────────
        instance = cls.__new__(cls)
        instance.engine = engine
        instance._config = cfg
        instance._fs_src = _fs_src
        instance._fs_in = _fs_in
        return instance

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sample_rates(
        cfg: dict,
        fs_src: int | None,
        fs_in: int | None,
    ) -> tuple[int, int]:
        """Resolve fs_src and fs_in with three-tier priority.

        Priority:
        1. Explicit kwargs (fs_src / fs_in) — highest priority.
        2. Training config: ``config["dataset"]``.
        3. Hardcoded fallback: fs_src=48000, fs_in=16000.

        Args:
            cfg:    Inner config dict (``yaml_dict["config"]``).
            fs_src: Explicit override for output sample rate, or ``None``.
            fs_in:  Explicit override for input sample rate, or ``None``.

        Returns:
            ``(fs_src, fs_in)`` as resolved integers.
        """
        _FALLBACK_FS_SRC = 48000
        _FALLBACK_FS_IN = 16000

        if fs_src is not None and fs_in is not None:
            return int(fs_src), int(fs_in)

        # Try training config
        try:
            _cfg_src = int(cfg["dataset"]["sample_rate_src"])
            _cfg_in = int(cfg["dataset"]["sample_rate_in"])
        except (KeyError, TypeError):
            _cfg_src = _FALLBACK_FS_SRC
            _cfg_in = _FALLBACK_FS_IN
            logger.info(
                f"Using hardcoded fallback sample rates "
                f"(fs_src={_FALLBACK_FS_SRC}, fs_in={_FALLBACK_FS_IN}). "
                f"Override with explicit kwargs if processing audio at different rates."
            )
        else:
            if _cfg_src != _FALLBACK_FS_SRC or _cfg_in != _FALLBACK_FS_IN:
                logger.info(
                    f"Using training config sample rates "
                    f"(fs_src={_cfg_src}, fs_in={_cfg_in}). "
                    f"Override with explicit kwargs if processing audio at different rates."
                )

        resolved_src = int(fs_src) if fs_src is not None else _cfg_src
        resolved_in = int(fs_in) if fs_in is not None else _cfg_in
        return resolved_src, resolved_in

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stft(self) -> dict:
        """STFT transform dict keyed by sample rate string.

        The dict maps ``str(fs)`` to a :class:`~tf_restormer.utils.util_stft.STFT`
        instance.  Use :meth:`get_stft` for a convenient lookup.

        Example::

            stft_fn = model.get_stft(16000)
            X = stft_fn(waveform, cplx=True)  # (1, F, T) complex
        """
        return self.engine.stft

    @property
    def istft(self) -> dict:
        """iSTFT transform dict keyed by sample rate string.

        Use :meth:`get_istft` for a convenient lookup.
        """
        return self.engine.istft

    def get_stft(self, fs: int):
        """Return the STFT module for the given sample rate.

        Falls back to the nearest available rate if *fs* is not exact.

        Args:
            fs: Sample rate in Hz.

        Returns:
            :class:`~tf_restormer.utils.util_stft.STFT` instance.
        """
        key = str(fs)
        if key in self.engine.stft:
            return self.engine.stft[key]
        nearest = min(self.engine.fs_list, key=lambda k: abs(int(k) - fs))
        return self.engine.stft[nearest]

    def get_istft(self, fs: int):
        """Return the iSTFT module for the given sample rate.

        Falls back to the nearest available rate if *fs* is not exact.

        Args:
            fs: Sample rate in Hz.

        Returns:
            :class:`~tf_restormer.utils.util_stft.iSTFT` instance.
        """
        key = str(fs)
        if key in self.engine.istft:
            return self.engine.istft[key]
        nearest = min(self.engine.fs_list, key=lambda k: abs(int(k) - fs))
        return self.engine.istft[nearest]

    # ------------------------------------------------------------------
    # Public inference methods
    # ------------------------------------------------------------------

    def process_waveform(
        self,
        waveform: torch.Tensor,
        fs_in: int | None = None,
        fs_out: int | None = None,
        **kwargs,
    ) -> dict:
        """Enhance a full waveform.

        Delegates to :meth:`EngineInfer.infer_session` which automatically
        chooses between single-pass and chunked overlap-add based on waveform
        length.

        Args:
            waveform: Input waveform, shape ``(L,)`` or ``(1, L)``.
            fs_in:    Input sample rate (Hz).  Defaults to ``self._fs_in``
                      (read from training config, typically 16000).
            fs_out:   Output sample rate (Hz).  Defaults to ``self._fs_src``
                      (training config output rate, typically 48000).
            **kwargs: Forwarded to :meth:`EngineInfer.infer_session`
                      (e.g., ``mode``, ``css_config``, ``show_progress``).

        Returns:
            dict with key ``"waveform"`` → enhanced tensor, shape ``(1, L_out)``.
        """
        return self.engine.infer_session(
            waveform,
            fs_in=fs_in,
            fs_out=fs_out,
            **kwargs,
        )

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        fs_in: int | None = None,
        fs_out: int | None = None,
    ) -> dict:
        """Enhance an audio file on disk.

        Reads the file, calls :meth:`process_waveform`, and optionally writes
        the enhanced output.

        Args:
            input_path:  Path to the input audio file (any format supported by
                         ``soundfile``).
            output_path: If provided, write the enhanced waveform to this path.
                         The output sample rate is ``fs_out`` (default:
                         ``self._fs_src``).
            fs_in:       Override the input sample rate detected from the file.
                         When ``None``, the file's actual sample rate is used and
                         resampling is performed if it differs from
                         ``self._fs_in``.
            fs_out:      Desired output sample rate.  Defaults to
                         ``self._fs_src``.

        Returns:
            dict with keys:
            - ``"waveform"`` → ``(1, L_out)`` enhanced tensor.
            - ``"sample_rate"`` → output sample rate (int).
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required for process_file(). "
                "Install with: pip install soundfile"
            ) from None

        input_path = Path(input_path)
        wav_np, orig_fs = sf.read(str(input_path), dtype="float32")

        # Stereo -> mono: keep first channel
        if wav_np.ndim > 1:
            wav_np = wav_np[:, 0]

        wav = torch.from_numpy(wav_np)  # (L,) float32

        # Resolve actual input rate
        _fs_in = fs_in if fs_in is not None else orig_fs

        # Resample to model's expected input rate if needed
        if _fs_in != self._fs_in:
            try:
                import torchaudio.functional as F_audio
            except ImportError:
                raise ImportError(
                    "torchaudio is required for sample rate conversion. "
                    "Install with: pip install torchaudio"
                ) from None
            logger.info(
                f"Resampling input from {_fs_in} Hz -> {self._fs_in} Hz."
            )
            wav = F_audio.resample(wav, _fs_in, self._fs_in)
            _fs_in = self._fs_in

        result = self.process_waveform(wav, fs_in=_fs_in, fs_out=fs_out)

        out_fs = fs_out if fs_out is not None else self._fs_src
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_np = result["waveform"].squeeze(0).cpu().numpy()
            sf.write(str(output_path), out_np, out_fs)
            logger.info(f"Saved enhanced audio: {output_path} ({out_fs} Hz)")

        result["sample_rate"] = out_fs
        return result

    @torch.inference_mode()
    def process_stft(
        self,
        stft_input: torch.Tensor,
        fs_in: int | None = None,
        fs_out: int | None = None,
    ) -> dict:
        """Run the model directly on a complex STFT input.

        For users who manage STFT computation themselves.  Accepts a complex
        tensor and returns both the enhanced STFT and the reconstructed
        waveform.

        Args:
            stft_input: Complex STFT tensor ``(1, F, T)`` or ``(F, T)``, or
                        stacked real/imag tensor ``(1, F, T, 2)``.
            fs_in:      Input sample rate used to select the correct STFT
                        key.  Defaults to ``self._fs_in``.
            fs_out:     Output sample rate used to select iSTFT and compute
                        ``out_F``.  Defaults to ``self._fs_src``.

        Returns:
            dict with keys:
            - ``"stft_out"`` → ``(1, F_out, T)`` complex64 tensor.
            - ``"waveform"`` → ``(1, L_out)`` float32 tensor.
        """
        _fs_in = fs_in if fs_in is not None else self._fs_in
        _fs_out = fs_out if fs_out is not None else self._fs_src

        # ── Normalize input shape ────────────────────────────────────────
        x = stft_input.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (F, T) -> (1, F, T)

        if x.is_complex():
            model_input = torch.view_as_real(x)   # (1, F, T, 2)
        elif x.dim() == 4 and x.shape[-1] == 2:
            model_input = x  # already stacked real/imag
        else:
            raise ValueError(
                f"Expected complex tensor (1, F, T) or real stacked tensor (1, F, T, 2), "
                f"got shape {stft_input.shape}, dtype {stft_input.dtype}"
            )

        # ── out_F computation (mirrors engine_infer.py L118-121) ─────────
        out_F = int(self.engine.frame_length * int(_fs_out) / 1000) // 2 + 1

        # ── iSTFT key selection with nearest-key fallback ────────────────
        istft_key = str(_fs_out)
        if istft_key not in self.engine.istft:
            istft_key = min(
                self.engine.fs_list, key=lambda k: abs(int(k) - _fs_out)
            )

        # ── Model forward (inside inference_mode context) ─────────────────
        comp = self.engine.model(model_input, out_F=out_F)   # (1, F_out, T, 2)
        out_cplx = torch.complex(comp[..., 0], comp[..., 1]) # (1, F_out, T)
        out_wav = self.engine.istft[istft_key](
            out_cplx, cplx=True, squeeze=False
        )  # (1, L_out)

        return {"stft_out": out_cplx, "waveform": out_wav}

    def create_session(
        self,
        streaming: bool = False,
        fs_in: int | None = None,
        fs_out: int | None = None,
        css_config: dict | None = None,
    ) -> InferenceSession:
        """Create a stateful :class:`InferenceSession` for chunk-by-chunk processing.

        Args:
            streaming: If ``True``, each :meth:`InferenceSession.feed_waveform`
                       call returns enhanced chunks immediately.  If ``False``
                       (batch mode), results accumulate for :meth:`InferenceSession.finalize`.
            fs_in:     Input sample rate for the session.  Defaults to
                       ``self._fs_in``.
            fs_out:    Output sample rate for the session.  Defaults to
                       ``self._fs_src``.
            css_config: Optional dict to override ``chunk_sec`` / ``overlap_sec``
                        on the underlying :class:`EngineInfer`.  Applied before
                        the session reads those values.

        Returns:
            :class:`InferenceSession` instance.
        """
        return InferenceSession(
            parent=self,
            streaming=streaming,
            fs_in=fs_in,
            fs_out=fs_out,
            css_config=css_config,
        )
