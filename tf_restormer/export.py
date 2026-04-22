"""Checkpoint export utility for TF_Restormer.

Copies trained checkpoints from the internal log/ directory structure
to a clean exportable location (default: tf_restormer/checkpoints/{config_stem}/).
Training code continues to write to log/ as before — this utility provides
a clean way to "publish" a trained checkpoint for library API consumption.

The exported checkpoint contains only the model weights (no optimizer state,
no profiling tensors), so it is significantly smaller than the raw training
checkpoint.

Note:
    ``huggingface_hub`` is an optional dependency (install with
    ``pip install tf-restormer[hub]`` or ``uv sync --extra hub``).
    Functions that require it (``upload_to_hub``, ``upload_all``,
    ``download_from_hub``) raise ``ImportError`` with an install hint
    when the package is not present.

CLI usage::

    # Export latest checkpoint for a config
    python tf_restormer/export.py --config baseline.yaml

    # Export and upload to HF Hub
    python tf_restormer/export.py --config baseline.yaml --upload --repo-id shinuh/tf-restormer-baseline

    # Upload all locally exported checkpoints
    python tf_restormer/export.py --upload-all

    # Download from HF Hub
    python tf_restormer/export.py --download --config baseline.yaml --repo-id shinuh/tf-restormer-baseline

Library usage::

    from tf_restormer.export import export_checkpoint
    ckpt_path = export_checkpoint("baseline.yaml")
"""
from __future__ import annotations

import argparse
import importlib.resources
import os
import shutil
from pathlib import Path

import torch
from loguru import logger

from tf_restormer._config import _VARIANT_MAP, load_config, resolve_config
from tf_restormer.inference import _strip_profiling_keys
from tf_restormer.utils.util_engine import _find_latest_checkpoint, _fix_compiled_state_dict, resolve_log_base


_DEFAULT_CKPT_HOME = Path(__file__).resolve().parent / "checkpoints"

# TF_Restormer is a single-variant project, but the map is kept for extensibility.
_SUPPORTED_VARIANTS = set(_VARIANT_MAP.keys())  # {"TF_Restormer"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_yaml_ext(config_name: str) -> str:
    """Return *config_name* with a ``.yaml`` extension, adding it if absent."""
    if not config_name.endswith(".yaml"):
        return config_name + ".yaml"
    return config_name


def _resolve_source_dir(
    config_name: str,
    variant: str = "TF_Restormer",
) -> Path:
    """Locate the latest checkpoint file inside the variant's log/ tree.

    The checkpoint directory follows the convention established by
    ``main_infer.py:L75-78``::

        <variant_dir>/log/log_{train_phase}_{config_name}/weights/

    where ``train_phase`` comes from the config YAML
    and ``config_name`` includes the ``.yaml`` extension.

    ``importlib.resources.files("tf_restormer.models.TF_Restormer")`` is used
    to resolve ``variant_dir`` because ``export.py`` lives at
    ``tf_restormer/export.py`` (one level above the model package), so a
    simple ``Path(__file__).parent``-relative path would point to the wrong
    directory.  This mirrors the pattern already used in ``_config.py``.

    Args:
        config_name: Config filename, e.g. ``"baseline.yaml"`` or ``"baseline"``.
                     The ``.yaml`` extension is appended if missing.
        variant:     Model variant key.  Currently only ``"TF_Restormer"``
                     is supported.

    Returns:
        Path to the latest ``epoch.{NNNN}.pth`` checkpoint file.

    Raises:
        KeyError: If *variant* is not recognized.
        FileNotFoundError: If the weights directory does not exist or
                           contains no ``epoch.*.pth`` files.
    """
    if variant not in _SUPPORTED_VARIANTS:
        raise KeyError(
            f"Unknown variant '{variant}'. Supported: {sorted(_SUPPORTED_VARIANTS)}"
        )

    config_name = _ensure_yaml_ext(config_name)

    # Resolve the model package directory via importlib.resources.
    # This is equivalent to os.path.dirname(os.path.abspath(main_infer.__file__))
    # but safe when called from export.py which lives one level higher.
    variant_dir = Path(
        str(importlib.resources.files("tf_restormer.models.TF_Restormer"))
    )

    # Load config to extract train_phase
    yaml_dict = load_config(variant, config_name)
    config = yaml_dict["config"]
    train_phase = config["train_phase"]

    # Mirror main_infer.py — use resolve_log_base for backward-compatible path discovery
    log_base = resolve_log_base(train_phase, config_name, str(variant_dir))
    chkp_dir = variant_dir / log_base / "weights"

    if not chkp_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found:\n"
            f"  {chkp_dir}\n"
            f"Train the model first or verify the config name."
        )

    result = _find_latest_checkpoint(str(chkp_dir))
    if result is None:
        raise FileNotFoundError(
            f"No epoch.*.pth checkpoints found in:\n"
            f"  {chkp_dir}\n"
            f"Train the model first."
        )

    ckpt_path, epoch = result
    logger.info(
        f"Found checkpoint: epoch {epoch:04d} at {ckpt_path}"
    )
    return Path(ckpt_path)


def _file_hash(path: Path | str) -> str:
    """Compute SHA-256 hex digest of *path* in 8 KB chunks."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file_path(ckpt_dir: Path | str) -> Path:
    """Return the path to the ``.upload_hash`` sentinel file."""
    return Path(ckpt_dir) / ".upload_hash"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_checkpoint(
    config_name: str,
    variant: str = "TF_Restormer",
    output_dir: Path | str | None = None,
) -> Path:
    """Export the latest trained checkpoint to a clean ``model.pt``.

    Locates the newest ``epoch.{NNNN}.pth`` checkpoint under the variant's
    ``log/`` tree, strips optimizer state and ptflops/thop profiling keys
    (``total_ops``, ``total_params``), and saves a compact checkpoint that
    contains only ``{"model_state_dict": ...}``.

    The config YAML is also copied alongside the exported checkpoint so that
    everything needed for inference lives in one directory.

    Note:
        When loading raw ``.pth`` training checkpoints directly (not the
        exported ``model.pt``), use ``model.load_state_dict(sd, strict=False)``
        because the profiling keys cause ``strict=True`` to raise an
        *unexpected key* error.

    Args:
        config_name: Config filename, e.g. ``"baseline.yaml"`` or
                     ``"baseline"``.  The ``.yaml`` extension is appended
                     automatically if missing.
        variant:     Model variant key (default ``"TF_Restormer"``).
        output_dir:  Destination directory.  Defaults to
                     ``tf_restormer/checkpoints/{config_stem}/``.

    Returns:
        Absolute path to the exported ``model.pt`` file.

    Raises:
        KeyError: If *variant* is not recognized.
        FileNotFoundError: If no trained checkpoint is found.

    Example::

        from tf_restormer.export import export_checkpoint
        dest = export_checkpoint("baseline.yaml")
        print(f"Exported to: {dest}")
    """
    config_name = _ensure_yaml_ext(config_name)
    config_stem = Path(config_name).stem  # "baseline"

    # Locate source checkpoint
    src_ckpt = _resolve_source_dir(config_name, variant)

    # Load checkpoint — prefer weights_only=True for safety
    try:
        ckpt = torch.load(str(src_ckpt), map_location="cpu", weights_only=True)
    except Exception as e:
        if "weights_only" not in str(e).lower():
            raise
        logger.warning(f"weights_only load failed, retrying without: {e}")
        ckpt = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)

    # Extract and clean model state dict
    state_dict = ckpt["model_state_dict"]

    # Fix keys from torch.compile() wrapping
    state_dict = _fix_compiled_state_dict(state_dict)

    # Strip ptflops / thop profiling keys (total_ops, total_params).
    state_dict = _strip_profiling_keys(state_dict)

    # Resolve output directory
    if output_dir is None:
        dest_dir = _DEFAULT_CKPT_HOME / config_stem
    else:
        dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Save exported checkpoint
    dest_ckpt = dest_dir / "model.pt"
    torch.save({"model_state_dict": state_dict}, str(dest_ckpt))
    logger.info(f"Exported checkpoint saved to: {dest_ckpt}")

    # Copy config YAML alongside the checkpoint
    src_yaml = resolve_config(variant, config_name)
    dest_yaml = dest_dir / "config.yaml"
    shutil.copy2(src_yaml, dest_yaml)
    logger.info(f"Config copied to: {dest_yaml}")

    return dest_ckpt


def upload_to_hub(
    config_name: str,
    repo_id: str | None = None,
    variant: str = "TF_Restormer",
    private: bool = True,
    token: str | None = None,
    force: bool = False,
) -> str | None:
    """Upload an exported checkpoint to the Hugging Face Hub.

    The checkpoint must already exist in ``checkpoints/`` (run
    :func:`export_checkpoint` first).  Skips the upload if the file hash
    matches the last upload (tracked via a ``.upload_hash`` sentinel).
    Pass ``force=True`` to bypass the hash check.

    Args:
        config_name: Config filename, e.g. ``"baseline.yaml"``.
        repo_id:     HF repo ID.  Auto-generated as
                     ``shinuh/tf-restormer-{config-slug}`` if ``None``.
        variant:     Model variant key (default ``"TF_Restormer"``).
        private:     Create a private repo (default ``True``).
        token:       HF API token.  Uses the cached token when ``None``.
        force:       Upload even if the checkpoint hasn't changed.

    Returns:
        URL of the HF repo, or ``None`` if the upload was skipped.

    Raises:
        FileNotFoundError: If the exported checkpoint does not exist.
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF Hub uploads.\n"
            "Install with: uv sync --extra hub  (or: pip install tf-restormer[hub])"
        )

    config_name = _ensure_yaml_ext(config_name)
    config_stem = Path(config_name).stem

    ckpt_path = _DEFAULT_CKPT_HOME / config_stem / "model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"No exported checkpoint at {ckpt_path}\n"
            f"Export first: python tf_restormer/export.py --config {config_name}"
        )

    # Hash-based skip
    current_hash = _file_hash(ckpt_path)
    hash_file = _hash_file_path(ckpt_path.parent)
    if (
        not force
        and hash_file.exists()
        and hash_file.read_text().strip() == current_hash
    ):
        logger.info(
            f"[SKIP] {config_stem} — checkpoint unchanged since last upload"
        )
        return None

    # Resolve config YAML
    yaml_path = resolve_config(variant, config_name)

    # Auto-generate repo_id
    if repo_id is None:
        slug = config_stem.lower().replace("_", "-")
        repo_id = f"shinuh/tf-restormer-{slug}"

    # Upload
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="model.pt",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=str(yaml_path),
        path_in_repo="config.yaml",
        repo_id=repo_id,
    )

    # Save hash sentinel
    hash_file.write_text(current_hash)

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Uploaded to: {url}")
    return url


def upload_all(
    variant: str = "TF_Restormer",
    private: bool = True,
    token: str | None = None,
    force: bool = False,
) -> dict[str, str | None]:
    """Upload all locally exported checkpoints to HF Hub.

    Iterates ``tf_restormer/checkpoints/*/model.pt`` and calls
    :func:`upload_to_hub` for each one.  Only uploads files that have
    changed since their last upload (unless ``force=True``).

    Args:
        variant: Model variant key (default ``"TF_Restormer"``).
        private: Create private repos (default ``True``).
        token:   HF API token.
        force:   Upload all regardless of hash state.

    Returns:
        ``{repo_id: url_or_None}`` for every checkpoint encountered.
        Failed uploads map to ``None``.
    """
    results: dict[str, str | None] = {}
    ckpt_root = _DEFAULT_CKPT_HOME

    if not ckpt_root.exists():
        logger.info("No checkpoints directory found; nothing to upload.")
        return results

    for config_dir in sorted(ckpt_root.iterdir()):
        if not config_dir.is_dir():
            continue
        model_pt = config_dir / "model.pt"
        if not model_pt.exists():
            continue

        config_name = config_dir.name + ".yaml"
        config_stem = config_dir.name
        slug = config_stem.lower().replace("_", "-")
        auto_repo_id = f"shinuh/tf-restormer-{slug}"

        try:
            url = upload_to_hub(
                config_name=config_name,
                variant=variant,
                private=private,
                token=token,
                force=force,
            )
            results[auto_repo_id] = url
        except Exception as exc:
            logger.error(f"[ERROR] {config_stem}: {exc}")
            results[auto_repo_id] = None

    uploaded = sum(1 for v in results.values() if v is not None)
    skipped = sum(1 for v in results.values() if v is None)
    logger.info(f"Done: {uploaded} uploaded, {skipped} skipped/failed")
    return results


def download_from_hub(
    repo_id: str,
    config_name: str | None = None,
    output_dir: Path | str | None = None,
    token: str | None = None,
) -> dict[str, Path]:
    """Download a checkpoint from HF Hub.

    Downloads ``model.pt`` (and optionally ``config.yaml``) from the given
    repo and copies them to ``tf_restormer/checkpoints/{config_stem}/``.

    Args:
        repo_id:     HF repo ID, e.g. ``"shinuh/tf-restormer-baseline"``.
        config_name: Config filename for local storage path.  When ``None``,
                     the config stem is inferred from the last segment of
                     *repo_id* (e.g. ``"tf-restormer-baseline"`` →
                     ``"baseline"``).
        output_dir:  Destination directory.  Defaults to
                     ``tf_restormer/checkpoints/{config_stem}/``.
        token:       HF API token.

    Returns:
        ``{"checkpoint": Path, "config": Path}`` pointing to local files.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF Hub downloads.\n"
            "Install with: uv sync --extra hub  (or: pip install tf-restormer[hub])"
        )

    # Infer config_stem from repo_id when config_name is not given
    if config_name is None:
        # e.g. "shinuh/tf-restormer-baseline" -> "baseline"
        last_segment = repo_id.split("/")[-1]  # "tf-restormer-baseline"
        prefix = "tf-restormer-"
        if last_segment.startswith(prefix):
            config_stem = last_segment[len(prefix):].replace("-", "_")
        else:
            config_stem = last_segment.replace("-", "_")
    else:
        config_name = _ensure_yaml_ext(config_name)
        config_stem = Path(config_name).stem

    # Download from Hub
    ckpt_cache = Path(
        hf_hub_download(repo_id=repo_id, filename="model.pt", token=token)
    )
    try:
        cfg_cache = Path(
            hf_hub_download(repo_id=repo_id, filename="config.yaml", token=token)
        )
    except Exception:
        logger.warning(
            "config.yaml not found in the Hub repo; skipping config download."
        )
        cfg_cache = None

    # Copy to output directory
    if output_dir is None:
        dest_dir = _DEFAULT_CKPT_HOME / config_stem
    else:
        dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_ckpt = dest_dir / "model.pt"
    shutil.copy2(ckpt_cache, dest_ckpt)

    if cfg_cache is not None:
        dest_cfg = dest_dir / "config.yaml"
        shutil.copy2(cfg_cache, dest_cfg)
    else:
        dest_cfg = dest_dir / "config.yaml"  # may not exist

    logger.info(
        f"Downloaded from: https://huggingface.co/{repo_id}\n"
        f"  Checkpoint: {dest_ckpt}\n"
        f"  Config:     {dest_cfg}"
    )
    return {"checkpoint": dest_ckpt, "config": dest_cfg}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export, upload, or download TF_Restormer checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Export latest checkpoint\n"
            "  python tf_restormer/export.py --config baseline.yaml\n"
            "\n"
            "  # Export then upload to HF Hub (auto-generate repo ID)\n"
            "  python tf_restormer/export.py --config baseline.yaml --upload\n"
            "\n"
            "  # Export then upload to a specific repo\n"
            "  python tf_restormer/export.py --config baseline.yaml --upload"
            " --repo-id shinuh/tf-restormer-baseline\n"
            "\n"
            "  # Upload all locally exported checkpoints to HF Hub\n"
            "  python tf_restormer/export.py --upload-all\n"
            "\n"
            "  # Download from HF Hub\n"
            "  python tf_restormer/export.py --download"
            " --repo-id shinuh/tf-restormer-baseline\n"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="CONFIG",
        help=(
            "Config filename used during training, e.g. 'baseline.yaml'. "
            "Required for export and upload."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for the exported checkpoint. "
            f"Defaults to tf_restormer/checkpoints/{{config_stem}}/."
        ),
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the latest checkpoint (default action when no flag is given).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Export checkpoint then upload to Hugging Face Hub.",
    )
    parser.add_argument(
        "--upload-all",
        action="store_true",
        help="Upload all locally exported checkpoints to HF Hub.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download a checkpoint from Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        metavar="REPO_ID",
        help=(
            "HF repo ID for upload/download "
            "(auto-generated as 'shinuh/tf-restormer-{config-slug}' if omitted)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if the checkpoint has not changed since last upload.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public HF repo (default is private).",
    )

    cli_args = parser.parse_args()

    if cli_args.upload_all:
        upload_all(private=not cli_args.public, force=cli_args.force)

    elif cli_args.download:
        if cli_args.repo_id is None and cli_args.config is None:
            parser.error("--download requires at least one of --repo-id or --config")
        result = download_from_hub(
            repo_id=cli_args.repo_id or "",
            config_name=cli_args.config,
            output_dir=cli_args.output,
        )
        print(f"Done: {result['checkpoint']}")

    elif cli_args.upload:
        if cli_args.config is None:
            parser.error("--upload requires --config")
        exported_path = export_checkpoint(
            config_name=cli_args.config,
            output_dir=cli_args.output,
        )
        url = upload_to_hub(
            config_name=cli_args.config,
            repo_id=cli_args.repo_id,
            private=not cli_args.public,
            force=cli_args.force,
        )
        print(f"Done: exported={exported_path}, hub={url}")

    else:
        # Default action: export
        if cli_args.config is None:
            parser.error("export requires --config")
        exported_path = export_checkpoint(
            config_name=cli_args.config,
            output_dir=cli_args.output,
        )
        print(f"Done: {exported_path}")
