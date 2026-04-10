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

_VARIANT_MAP = {
    "TF_Restormer": "tf_restormer.models.TF_Restormer",
}


def resolve_config(variant: str, config_name: str) -> str:
    """Resolve a config alias to an absolute YAML file path.

    Args:
        variant: One of "TF_Restormer".
        config_name: YAML filename (e.g. "baseline.yaml").

    Returns:
        Absolute path to the YAML config file (valid for editable installs).

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If variant is not recognized.
    """
    package = _VARIANT_MAP[variant]
    ref = importlib.resources.files(package).joinpath("configs", config_name)
    if not ref.is_file():
        raise FileNotFoundError(f"Config not found: {variant}/{config_name}")
    return str(ref)


def resolve_testsets(variant: str) -> str:
    """Resolve the testsets.yaml path for a given variant.

    Args:
        variant: One of "TF_Restormer".

    Returns:
        Absolute path to configs/testsets.yaml (valid for editable installs).

    Raises:
        FileNotFoundError: If testsets.yaml does not exist.
        KeyError: If variant is not recognized.
    """
    package = _VARIANT_MAP[variant]
    ref = importlib.resources.files(package).joinpath("configs", "testsets.yaml")
    if not ref.is_file():
        raise FileNotFoundError(f"testsets.yaml not found for variant: {variant}")
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
    except FileNotFoundError:
        testset_defs = {}  # No testsets.yaml → assume inline definitions (backward compat)

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
