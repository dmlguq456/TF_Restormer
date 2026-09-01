"""Independent arithmetic/integrity checks for the collapse diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forensics", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    forensic = json.loads(args.forensics.read_text())
    probe = json.loads(args.probe.read_text())
    intervals = {(row["start_epoch"], row["end_epoch"]): row for row in forensic["intervals"]}
    e19_20 = intervals[(19, 20)]
    output = probe["mean_hf_lf_db"]["outputs"]
    effect = probe["actual_update_effect"]
    checks = {
        "pre20_variable_rate_fingerprint": forensic["inference"]["variable_rate_path_active_before_epoch20"],
        "epoch20_steps_match_urgent_scp": forensic["inference"]["epoch20_generator_steps_match_urgent_scp"],
        "two_discriminator_updates_per_batch": forensic["inference"]["epoch20_discriminator_two_updates_per_batch"],
        "baseline_loss_weights_exact": forensic["inference"]["epoch20_uses_baseline_loss_weights"],
        "urgent_snapshot_weights_rejected": forensic["inference"]["epoch20_does_not_use_preserved_urgent_loss_weights"],
        "resume_command_uses_baseline_config": forensic["inference"]["resume_command_uses_baseline_config"],
        "urgent_eval_preceded_baseline_resume": forensic["inference"]["urgent_eval_preceded_baseline_resume"],
        "post_resume_epochs_are_1000_steps": forensic["inference"]["post_resume_epochs_are_1000_steps"],
        "hf_delta_recomputed": abs((output["epoch20"] - output["epoch19"]) - effect["hf_lf_delta_db"]) < 1e-9,
        "probe_has_three_samples": len(probe["samples"]) == 3,
        "estimator_delta_nonzero": probe["actual_estimator_delta_l2"] > 0,
        "finite_alignments": all(abs(value) <= 1.000001 for value in probe["gradient_alignment"].values()),
        "epoch20_generator_steps_1000": e19_20["generator_steps"] == 1000,
    }
    result = {"schema_version": 1, "checks": checks, "pass": all(checks.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
