"""Independent arithmetic and historical-component verification for the probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


TAGS = {
    "se": "Loss_se/Train/Loss_se",
    "time": "Loss_time/Train/Loss_time",
    "ssl": "Loss_rep/Train/Loss_rep",
    "gan": "Loss_G_adv/Train/Loss_G_adv",
    "fm": "Loss_G_fm/Train/Loss_G_fm",
    "pesq": "Loss_pesq/Train/Loss_pesq",
}
WEIGHTS = {"se": 1.0, "time": 0.0, "ssl": 100.0, "gan": 0.001, "fm": 0.1, "pesq": 0.0001}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tbparse(path: Path):
    spec = importlib.util.spec_from_file_location("tbparse_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--tbparse", type=Path, required=True)
    parser.add_argument("--epoch19", type=Path, required=True)
    parser.add_argument("--epoch20", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--components-output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    rows = metrics["rows"]
    rates = sorted(int(rate) for rate in metrics["rate_weights"])
    rate_weights = {int(k): v for k, v in metrics["rate_weights"].items()}

    independent_means = {
        key: {
            fs: float(np.mean([row[key] for row in rows if row["fs"] == fs]))
            for fs in rates
        }
        for key in ("affected_fixed", "affected_buggy")
    }
    fixed = sum(rate_weights[fs] * independent_means["affected_fixed"][fs] for fs in rates)
    buggy = sum(rate_weights[fs] * independent_means["affected_buggy"][fs] for fs in rates)
    delta = buggy - fixed
    gap = metrics["observed_epoch19_to_20_gap"]
    reference = independent_means["affected_fixed"][48000]
    fixed_dev = abs(fixed - reference)
    buggy_dev = abs(buggy - reference)
    closeness = buggy_dev / max(fixed_dev, 1e-12)
    primary = delta > 0 and delta / gap >= 0.5 and closeness >= 2.0

    tbparse = load_tbparse(args.tbparse)
    scalar_rows = defaultdict(list)
    for event_file in glob.glob(str(args.events / "**/events.out.tfevents.*"), recursive=True):
        relative_dir = os.path.relpath(Path(event_file).parent, args.events)
        for step, tag, value in tbparse.scalars(event_file):
            key = tag if relative_dir == "." else f"{relative_dir}/{tag}"
            scalar_rows[key].append((step, value))
    components = {}
    for name, tag in TAGS.items():
        values = [value for step, value in scalar_rows[tag] if step == 20]
        if len(values) != 1:
            raise RuntimeError(f"expected one epoch-20 {tag} value, found {len(values)}")
        components[name] = values[0]

    weighted_components = {name: components[name] * WEIGHTS[name] for name in components}
    reconstructed = sum(weighted_components.values())
    unaffected = sum(weighted_components[name] for name in ("gan", "fm", "pesq"))
    epoch19 = torch.load(args.epoch19, map_location="cpu", weights_only=False, mmap=True)
    epoch20 = torch.load(args.epoch20, map_location="cpu", weights_only=False, mmap=True)
    historical = {
        "epoch19_train_loss": float(epoch19["train_loss"]),
        "epoch20_train_loss": float(epoch20["train_loss"]),
        "observed_gap": float(epoch20["train_loss"] - epoch19["train_loss"]),
        "raw_components_epoch20": components,
        "weights": WEIGHTS,
        "weighted_components_epoch20": weighted_components,
        "reconstructed_epoch20_total": reconstructed,
        "reconstruction_error": reconstructed - float(epoch20["train_loss"]),
        "unaffected_gan_fm_pesq": unaffected,
        "unaffected_fraction_of_observed_gap": unaffected / gap,
    }

    checks = {
        "row_count_12": len(rows) == 12,
        "rates_complete": all(sum(row["fs"] == fs for row in rows) == 3 for fs in rates),
        "checkpoint_hash_matches": file_sha256(args.epoch19) == metrics["checkpoint_sha256"],
        "model_load_exact": metrics["model_load"] == {"missing": [], "unexpected": []},
        "48k_semantics_identical": all(
            row["affected_fixed"] == row["affected_buggy"]
            for row in rows if row["fs"] == 48000
        ),
        "expected_fixed_recomputed": abs(fixed - metrics["decision"]["expected_affected_fixed"]) < 1e-12,
        "expected_buggy_recomputed": abs(buggy - metrics["decision"]["expected_affected_buggy"]) < 1e-12,
        "decision_recomputed": primary == metrics["decision"]["primary_cause"],
        "historical_gap_matches": abs(historical["observed_gap"] - gap) < 1e-12,
    }
    verification = {
        "schema_version": 1,
        "checks": checks,
        "passed": all(checks.values()),
        "independent_decision": {
            "bug_delta": delta,
            "explained_fraction": delta / gap,
            "closeness_improvement": closeness,
            "primary_cause": primary,
        },
        "external_write_gate": "closed-primary-cause-false" if not primary else "open",
    }
    args.components_output.write_text(json.dumps(historical, indent=2, sort_keys=True) + "\n")
    args.verification_output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
