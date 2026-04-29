"""config_override.py — How to override config values at load time.

This example demonstrates every supported way to pass a config to
SEInference.from_pretrained():

  Pattern 1 — Config name string (package-bundled YAML)
  Pattern 2 — Absolute path to a custom YAML file
  Pattern 3 — Dict (full YAML dict or inner config dict) with modifications
  Pattern 4 — Override output sample rate via fs_src kwarg
  Pattern 5 — Hugging Face Hub loading

Run this file to see what each pattern prints; no real inference is
performed (no checkpoint is required — the model build is skipped once
from_pretrained() raises FileNotFoundError for a missing checkpoint).
The patterns are shown as annotated code blocks intended to be copied.

IMPORTANT — dict config must be complete:
    The current API does NOT support partial dict merge.  When you pass a
    dict, every key that the model needs must be present.  Passing only the
    keys you want to change will cause a KeyError or TypeError at model
    construction time.

    Safe pattern: load the full config via _config.load_config(), mutate the
    specific value, then pass the whole dict to from_pretrained().
"""
from __future__ import annotations

import sys


# ---------------------------------------------------------------------------
# Pattern 1 — Config name (package-bundled YAML)
# ---------------------------------------------------------------------------
def pattern1_config_name() -> None:
    """Load using a bundled config name.  Most common usage."""
    print("=== Pattern 1: config name string ===")
    print(
        "model = SEInference.from_pretrained(\n"
        "    config='baseline.yaml',\n"
        "    checkpoint_path='path/to/checkpoints/baseline/',\n"
        "    device='cuda',\n"
        ")"
    )

    # Actual call (requires checkpoint to be present):
    #
    #   from tf_restormer import SEInference
    #   model = SEInference.from_pretrained(
    #       config="baseline.yaml",
    #       checkpoint_path="path/to/checkpoints/baseline/",
    #       device="cuda",
    #   )
    print()


# ---------------------------------------------------------------------------
# Pattern 2 — Absolute path to a custom YAML file
# ---------------------------------------------------------------------------
def pattern2_absolute_yaml_path() -> None:
    """Load from an arbitrary YAML file outside the package."""
    print("=== Pattern 2: absolute YAML path ===")
    print(
        "model = SEInference.from_pretrained(\n"
        "    config='/path/to/my_custom_config.yaml',\n"
        "    checkpoint_path='/path/to/model.pt',\n"
        "    device='cuda',\n"
        ")"
    )

    # Actual call:
    #
    #   from tf_restormer import SEInference
    #   model = SEInference.from_pretrained(
    #       config="/path/to/my_custom_config.yaml",
    #       checkpoint_path="/path/to/model.pt",
    #       device="cuda",
    #   )
    print()


# ---------------------------------------------------------------------------
# Pattern 3 — Dict (load YAML, modify, pass full dict)
# ---------------------------------------------------------------------------
def pattern3_dict_override() -> None:
    """Modify a config value programmatically, then pass the full dict.

    Two sub-forms are accepted:
      a) Full yaml_dict (has top-level "config" key) — _from_pretrained_impl
         extracts yaml_dict["config"] automatically.
      b) Inner config dict only (has "model", "dataset", ... keys directly).

    Either way, the dict must be COMPLETE.  Missing keys will crash at
    model construction.
    """
    print("=== Pattern 3: dict config with modifications ===")

    from tf_restormer._config import load_config  # noqa: PLC0415

    # Load the full YAML dict (the outer wrapper that contains "config" key)
    yaml_dict = load_config("TF_Restormer", "baseline.yaml")

    # Inspect current value
    current_val = yaml_dict["config"]["model"].get("num_blocks", "<not set>")
    print(f"  Original model.num_blocks = {current_val}")

    # --- Sub-form A: mutate the full yaml_dict and pass it ---
    yaml_dict["config"]["model"]["num_blocks"] = 2  # example override
    print("  Modified model.num_blocks = 2")

    # Actual call (requires checkpoint):
    #
    #   from tf_restormer import SEInference
    #   model = SEInference.from_pretrained(
    #       config=yaml_dict,          # full dict: {"config": {...}, ...}
    #       checkpoint_path="path/to/checkpoints/baseline/",
    #       device="cuda",
    #   )

    print(
        "\n  from_pretrained(config=yaml_dict, ...)  # full yaml_dict form\n"
    )

    # --- Sub-form B: pass only the inner config dict ---
    inner_cfg = yaml_dict["config"]  # {"model": {...}, "dataset": {...}, ...}

    # Actual call (requires checkpoint):
    #
    #   model = SEInference.from_pretrained(
    #       config=inner_cfg,          # inner dict: {"model": {...}, ...}
    #       checkpoint_path="path/to/checkpoints/baseline/",
    #       device="cuda",
    #   )

    print(
        "  from_pretrained(config=inner_cfg, ...)  # inner config dict form\n"
    )


# ---------------------------------------------------------------------------
# Pattern 4 — Override output sample rate via fs_src kwarg
# ---------------------------------------------------------------------------
def pattern4_sample_rate_override() -> None:
    """Override the output sample rate without touching the config dict.

    fs_src : sample rate the model produces at its output (e.g. 48000 Hz).
             Overrides the value read from the training config.

    Note: fs_in is no longer a from_pretrained() argument.  The input
    sample rate is supplied per-call (process_waveform, create_session, etc.)
    and the engine uses nearest-key STFT lookup, so no manual resampling is
    required.
    """
    print("=== Pattern 4: output sample rate override via fs_src kwarg ===")
    print(
        "model = SEInference.from_pretrained(\n"
        "    config='baseline.yaml',\n"
        "    checkpoint_path='path/to/checkpoints/baseline/',\n"
        "    device='cuda',\n"
        "    fs_src=16000,  # override: model produces 16 kHz output\n"
        ")"
    )

    # Actual call:
    #
    #   from tf_restormer import SEInference
    #   model = SEInference.from_pretrained(
    #       config="baseline.yaml",
    #       checkpoint_path="path/to/checkpoints/baseline/",
    #       device="cuda",
    #       fs_src=16000,
    #   )
    print()


# ---------------------------------------------------------------------------
# Pattern 5 — Hugging Face Hub
# ---------------------------------------------------------------------------
def pattern5_hf_hub() -> None:
    """Download checkpoint directly from the Hugging Face Hub.

    Pass a HF repo ID (owner/name) as checkpoint_path.  The model weights
    and config.yaml are downloaded automatically via huggingface_hub.

    Requires: uv sync --extra hub  (or: pip install tf-restormer[hub])
    """
    print("=== Pattern 5: Hugging Face Hub ===")
    print(
        "model = SEInference.from_pretrained(\n"
        "    checkpoint_path='shinuh/tf-restormer-baseline',\n"
        "    device='cuda',\n"
        ")"
    )

    # Actual call:
    #
    #   from tf_restormer import SEInference
    #   model = SEInference.from_pretrained(
    #       checkpoint_path="shinuh/tf-restormer-baseline",
    #       device="cuda",
    #   )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(__doc__)
    print("-" * 60)
    pattern1_config_name()
    pattern2_absolute_yaml_path()

    try:
        pattern3_dict_override()
    except Exception as exc:
        print(f"[Pattern 3 skipped — config not found in package: {exc}]", file=sys.stderr)
        print()

    pattern4_sample_rate_override()
    pattern5_hf_hub()

    print("All patterns printed.  Copy the relevant snippet into your project.")


if __name__ == "__main__":
    main()
