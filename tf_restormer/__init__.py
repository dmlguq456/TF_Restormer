"""tf_restormer -- Time-Frequency Restormer speech enhancement package."""

__all__ = ["SEInference", "InferenceSession"]


def __getattr__(name):
    """Lazy import: torch and inference modules are loaded only on first access."""
    if name in ("SEInference", "InferenceSession"):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorch is required but not installed. "
                "Install with an accelerator extra:\n"
                "  uv sync --extra cu126    # CUDA 12.6\n"
                "  uv sync --extra cpu      # CPU-only"
            ) from None
        from tf_restormer.inference import SEInference, InferenceSession
        globals()["SEInference"] = SEInference
        globals()["InferenceSession"] = InferenceSession
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
