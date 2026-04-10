"""
tf_restormer.utils.metrics
==========================
Registry-based metric dispatcher with lazy imports and device-aware model caching.

Usage
-----
    from tf_restormer.utils.metrics import compute_metric, list_metrics

    score = compute_metric('pesq', estim_wav, target_wav, fs=16000)
    keys  = list_metrics()
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Metric registry
# Maps: key -> (submodule_name_relative_to_this_package, function_name)
# ---------------------------------------------------------------------------
_METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    # intrusive (metrics-intrusive)
    'pesq':        ('intrusive',    'compute_pesq'),
    'stoi':        ('intrusive',    'compute_stoi'),
    'sdr':         ('intrusive',    'compute_sdr'),
    'lsd':         ('intrusive',    'compute_lsd'),
    'mcd':         ('intrusive',    'compute_mcd'),
    'composite':   ('intrusive',    'compute_composite'),
    # nonintrusive (metrics-nonintrusive)
    'dnsmos':      ('nonintrusive', 'compute_dnsmos'),
    'dnsmos_sig':  ('nonintrusive', 'compute_dnsmos_sig'),
    'dnsmos_bak':  ('nonintrusive', 'compute_dnsmos_bak'),
    'nisqa':       ('nonintrusive', 'compute_nisqa'),
    # neural (metrics-neural)
    'utmos':       ('neural',       'compute_utmos'),
    'wvmos':       ('neural',       'compute_wvmos'),
    # semantic (metrics-semantic)
    'bleu':        ('semantic',     'compute_speechbleu'),
    'bertscore':   ('semantic',     'compute_speechbertscore'),
    'tokendist':   ('semantic',     'compute_speechtokendist'),
    # asr (metrics-neural — shares transformers dep)
    'wer_whisper': ('asr',          'compute_wer_whisper'),
    'wer_w2v':     ('asr',          'compute_wer_w2v'),
    'cer_whisper': ('asr',          'compute_cer_whisper'),
}

# Human-readable pip extra hint per submodule
_EXTRA_MAP: dict[str, str] = {
    'intrusive':    'metrics-intrusive',
    'nonintrusive': 'metrics-nonintrusive',
    'neural':       'metrics-neural',
    'semantic':     'metrics-semantic',
    'asr':          'metrics-neural',
}


# ---------------------------------------------------------------------------
# Device-aware model cache
# ---------------------------------------------------------------------------

class _ModelCache:
    """Device-aware model cache for GPU-based metric scorers.

    Stores ``{(model_key, device_str): model_instance}`` pairs so that:
    - Different devices get separate instances (no stale cache on device change).
    - DDP: each rank has its own process / its own ``_model_cache`` — no conflicts.
    - Memory release: call ``_model_cache.clear()`` after all testsets complete
      (from the eval loop exit in ``main_infer.py``, NOT from inside run_eval).
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}

    def get_or_create(self, key: str, device: str, factory: Callable) -> Any:
        """Return cached model or create via ``factory(device)``."""
        cache_key = (key, str(device))
        if cache_key not in self._cache:
            self._cache[cache_key] = factory(device)
        return self._cache[cache_key]

    def clear(self, key: str | None = None) -> None:
        """Clear all entries, or entries matching a specific model key."""
        if key is None:
            self._cache.clear()
        else:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != key}


# Module-level singleton — one per process (DDP-safe).
_model_cache = _ModelCache()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metric(key: str, estim, target=None, fs: int = 16000, **kwargs):
    """Dispatch to the appropriate ``compute_*`` function by registry key.

    Parameters
    ----------
    key:    Metric key (e.g. ``'pesq'``, ``'dnsmos'``).
    estim:  Estimated/enhanced signal (torch.Tensor or np.ndarray).
    target: Reference/clean signal. Required for intrusive metrics.
    fs:     Sampling rate in Hz (default 16000).
    **kwargs: Forwarded to the underlying ``compute_*`` function.

    Returns
    -------
    float or tuple — whatever the underlying function returns.

    Raises
    ------
    KeyError:       If ``key`` is not in the registry.
    ImportError:    If a required optional dependency is not installed.
    """
    if key not in _METRIC_REGISTRY:
        available = list(_METRIC_REGISTRY.keys())
        raise KeyError(
            f"Unknown metric key '{key}'. "
            f"Available keys: {available}"
        )

    submodule_name, func_name = _METRIC_REGISTRY[key]
    module = importlib.import_module(
        f'tf_restormer.utils.metrics.{submodule_name}'
    )
    func = getattr(module, func_name)

    # Intrusive metrics need target; non-intrusive do not.
    if target is not None:
        return func(estim, target, fs, **kwargs)
    else:
        return func(estim, fs, **kwargs)


def list_metrics() -> list[str]:
    """Return the list of all registered metric keys."""
    return list(_METRIC_REGISTRY.keys())


def get_extra_name(key: str) -> str:
    """Return the pip extra name required to install the given metric key.

    Example
    -------
        >>> get_extra_name('pesq')
        'metrics-intrusive'
        >>> get_extra_name('dnsmos')
        'metrics-nonintrusive'
    """
    if key not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric key '{key}'.")
    submodule_name, _ = _METRIC_REGISTRY[key]
    return _EXTRA_MAP.get(submodule_name, 'unknown')
