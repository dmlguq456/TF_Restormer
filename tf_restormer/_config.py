"""Config name resolution -- maps alias strings to YAML file paths.

Note:
    This module assumes editable install (``pip install -e .``).
    The returned path string points to the real filesystem location of the YAML file.
    For non-editable (wheel) installs, the Traversable.__str__() may return
    a path inside a zip archive that cannot be opened with plain ``open()``.
    If wheel support is needed in the future, callers should use
    ``importlib.resources.as_file()`` context manager around the full
    config-loading lifecycle instead.

Design note:
    This module is larger than SR_CorrNet's _config.py (which has only
    resolve_config) because TF_Restormer uses a testset catalog pattern
    (testsets.yaml) that requires load_testsets/load_config/expand_env_vars.
    These functions are cohesive -- splitting is not warranted at this scale.
"""
from __future__ import annotations

import importlib.resources
import os
import re

import yaml
from loguru import logger

_VARIANT_MAP = {
    "TF_Restormer": "tf_restormer.models.TF_Restormer",
}


def _normalize_variant(variant: str) -> str:
    """Resolve a case-insensitive variant alias to the canonical key.

    Examples: "tf_restormer", "TF_RESTORMER", "tf-restormer" → "TF_Restormer".

    Args:
        variant: Any case or separator variant of a supported model name.

    Returns:
        The canonical key present in ``_VARIANT_MAP``.

    Raises:
        KeyError: If no matching entry is found.
    """
    if variant in _VARIANT_MAP:
        return variant
    # Explicit alias table — exact tokens only. Avoids silent acceptance of
    # typos such as "TFRest_Or_Mer" that a naive separator-stripping rule
    # would canonicalize to the same key as "TF_Restormer".
    _VARIANT_ALIASES = {
        "tf_restormer": "TF_Restormer",
        "tfrestormer": "TF_Restormer",
        "tf-restormer": "TF_Restormer",
        "tf_restormer".upper(): "TF_Restormer",
        "TFRestormer": "TF_Restormer",
    }
    key = _VARIANT_ALIASES.get(variant) or _VARIANT_ALIASES.get(variant.lower())
    if key is None:
        raise KeyError(
            f"Unknown variant {variant!r}. Allowed aliases: "
            f"{sorted(set(_VARIANT_ALIASES.values()) | set(_VARIANT_ALIASES.keys()))}"
        )
    return key


def resolve_config(variant: str, config_name: str) -> str:
    """Resolve a config alias to an absolute YAML file path.

    Args:
        variant: One of "TF_Restormer" (case-insensitive).
        config_name: YAML filename (e.g. "baseline.yaml") or absolute path.

    Returns:
        Absolute path to the YAML config file (valid for editable installs).

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If variant is not recognized.
    """
    if os.path.isabs(config_name):
        if not os.path.isfile(config_name):
            raise FileNotFoundError(
                f"Config file not found at absolute path: {config_name!r}"
            )
        return config_name
    variant = _normalize_variant(variant)
    package = _VARIANT_MAP[variant]
    ref = importlib.resources.files(package).joinpath("configs", config_name)
    if not ref.is_file():
        raise FileNotFoundError(
            f"Config not found in package {package!r}: configs/{config_name}. "
            "Pass an absolute path or use one of the bundled configs "
            "(baseline.yaml, streaming.yaml)."
        )
    return str(ref)


def resolve_testsets(variant: str) -> str:
    """Resolve the package-bundled testsets.yaml path.

    Uses ``importlib.resources`` to locate the file inside the model
    variant package (mirroring :func:`resolve_config`). Returns a string
    path that, on editable installs, points to the real filesystem; on
    wheel installs, the Traversable string may point inside a zip and
    cannot be opened with plain ``open()``. For wheel support, callers
    should wrap the load lifecycle in ``importlib.resources.as_file()``.

    Args:
        variant: One of "TF_Restormer".

    Returns:
        Path string to ``configs/testsets.yaml`` inside the variant package.

    Raises:
        FileNotFoundError: If testsets.yaml does not exist in the package.
        KeyError: If ``variant`` is not in ``_VARIANT_MAP``.
    """
    variant = _normalize_variant(variant)
    package = _VARIANT_MAP[variant]
    ref = importlib.resources.files(package).joinpath("configs", "testsets.yaml")
    if not ref.is_file():
        raise FileNotFoundError(f"testsets.yaml not found at: {ref}")
    return str(ref)


def expand_env_vars(value: str | None) -> str | None:
    """Expand ``${VAR}`` placeholders in *value* using environment variables.

    Expands ``${VAR}`` from environment variables (shell/export).
    The ``.env`` file is no longer loaded automatically.

    Args:
        value: A string that may contain ``${VAR}`` patterns, or ``None``.

    Returns:
        The expanded string, or ``None`` if *value* is ``None``.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if value is None:
        return None

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        result = os.getenv(var_name)
        if result is None:
            raise ValueError(
                f"Environment variable '${{{var_name}}}' is not set. "
                "Set it via `export VAR=...` in your shell, "
                "or use direct paths (db_root / rir_dir) in your YAML config."
            )
        return result

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def load_testsets(variant: str) -> dict:
    """Load and return the parsed testsets YAML for *variant*.

    Convenience wrapper around :func:`resolve_testsets` + ``yaml.safe_load``.

    Args:
        variant: One of "TF_Restormer".

    Returns:
        Parsed YAML content as a Python dict.

    Raises:
        FileNotFoundError: If testsets.yaml does not exist.
        KeyError: If variant is not recognized.
    """
    variant = _normalize_variant(variant)
    path = resolve_testsets(variant)
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(variant: str, config_name: str) -> dict:
    """Load a model config YAML and merge testset definitions from testsets.yaml.

    Resolves the config path, parses YAML, then merges testset catalog entries
    into ``config["dataset_test"]``.  Testset fields defined inline in the model
    config take priority over catalog defaults on a per-field basis.

    Args:
        variant: One of "TF_Restormer".
        config_name: YAML filename (e.g. "baseline.yaml") or absolute path.

    Returns:
        The full parsed ``yaml_dict`` (includes top-level ``config`` key),
        matching the original YAML config loading convention.

    Note:
        ``${VAR}`` placeholders are NOT expanded here.  Expansion is deferred to
        ``EvalDataset.__init__`` (Phase 5.7) to avoid crashing training-only runs
        where eval dataset env vars (e.g. ``VCTK_DEMAND_DB_ROOT``) are not set.
    """
    variant = _normalize_variant(variant)
    if os.path.isabs(config_name) and os.path.isfile(config_name):
        # NOTE: Even with absolute config paths, testsets.yaml is loaded from the
        # package. Inline testset definitions in the config take precedence over
        # catalog entries (deep merge, per-field basis).
        yaml_path = config_name
    else:
        yaml_path = resolve_config(variant, config_name)

    # safe_load supports anchors/aliases (PyYAML 5.1+)
    with open(yaml_path, encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f)

    if not isinstance(yaml_dict, dict) or "config" not in yaml_dict:
        raise ValueError(
            f"Invalid config file: {yaml_path!r} -- "
            "expected a YAML mapping with a top-level 'config' key."
        )
    config = yaml_dict["config"]

    try:
        testset_defs = load_testsets(variant)
    except FileNotFoundError as exc:
        logger.warning(
            f"testsets.yaml not found for variant {variant!r}: {exc}. "
            "Catalog merge skipped — only inline 'dataset_test' entries from "
            "the config will be available. eval/infer paths that look up "
            "testset definitions by key will fail with a friendly RuntimeError."
        )
        testset_defs = {}

    if testset_defs:
        dt = config.setdefault("dataset_test", {})
        for key, val in testset_defs.items():
            if key in dt and isinstance(dt[key], dict):
                # Shallow merge — assumes testset fields are flat (str, int, list).
                # Nested dicts require recursive merge.
                # Catalog provides defaults; inline config overrides.
                merged = {**val, **dt[key]}
                dt[key] = merged
            elif key not in dt:
                dt[key] = val
            # else: non-dict inline value (control key) — leave as-is

    # NOTE: ${VAR} expansion is NOT done here. It is deferred to
    # EvalDataset.__init__ (Phase 5.7) to avoid crashing training-only
    # runs where eval env vars (VCTK_DEMAND_DB_ROOT etc.) are not set.

    return yaml_dict


def apply_cli_gpuid(config: dict, args: object) -> None:
    """If ``--gpuid`` was supplied on the CLI, mutate ``config['engine']['gpuid']``.

    This allows the CLI flag to override whatever is set in the YAML without
    callers having to duplicate the same two lines in every entry point.

    Args:
        config: Full experiment config dict (mutated in-place).
        args:   argparse.Namespace from the CLI (or any object with a
                ``gpuid`` attribute).  A missing or empty attribute is a no-op.
    """
    cli_gpuid = getattr(args, "gpuid", None)
    if cli_gpuid is None or str(cli_gpuid) == "":
        return
    original = config.get("engine", {}).get("gpuid")
    if original != str(cli_gpuid):
        logger.info(
            f"Overriding YAML engine.gpuid={original!r} with CLI --gpuid={cli_gpuid!r}"
        )
    config.setdefault("engine", {})["gpuid"] = str(cli_gpuid)
