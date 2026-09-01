"""Reconstruct the epoch-19/20 resume conditions from preserved artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import yaml

from tf_restormer import _config
from tf_restormer.models.TF_Restormer.model import Model


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def parameter_steps(checkpoint: dict, names: list[str]) -> dict[str, int]:
    optimizer = checkpoint["optimizer_state_dict"]
    param_ids = optimizer["param_groups"][0]["params"]
    result = {}
    for index, param_id in enumerate(param_ids):
        state = optimizer["state"].get(param_id, {})
        if "step" in state:
            result[names[index]] = int(state["step"])
    return result


def scalar(path: Path, key: str) -> float:
    data = json.loads(path.read_text())
    return float(data["raw_components"][key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch20-components", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()

    cfg = _config.load_config("TF_Restormer", "baseline.yaml")["config"]
    names = [name for name, _ in Model(**cfg["model"]).named_parameters()]
    log = args.repo / "tf_restormer/models/TF_Restormer/log/log_adversarial_to48k_baseline.yaml"

    checkpoint_epochs = (16, 19, 20, 21, 22, 23, 24, 25)
    generator = {
        epoch: load(log / "weights" / f"epoch.{epoch:04d}.pth")
        for epoch in checkpoint_epochs
    }
    discriminator = {
        epoch: load(log / "weights_D" / f"epoch.{epoch:04d}.pth")
        for epoch in checkpoint_epochs
    }
    steps = {epoch: parameter_steps(checkpoint, names) for epoch, checkpoint in generator.items()}
    common_name = "estimator.mag.out.0.weight"
    pad_name = "up.mask_token"

    def interval(start: int, end: int) -> dict:
        common = steps[end][common_name] - steps[start][common_name]
        pad = steps[end][pad_name] - steps[start][pad_name]
        d_start = next(iter(discriminator[start]["optimizer_state_dict"]["state"].values()))["step"]
        d_end = next(iter(discriminator[end]["optimizer_state_dict"]["state"].values()))["step"]
        return {
            "start_epoch": start,
            "end_epoch": end,
            "generator_steps": common,
            "padding_path_steps": pad,
            "padding_path_fraction": pad / common,
            "discriminator_optimizer_steps": int(d_end) - int(d_start),
        }

    components = json.loads(args.epoch20_components.read_text())
    raw = components["raw_components_epoch20"]
    baseline_weights = {"se": 1.0, "ssl": 100.0, "gan": 0.001, "fm": 0.1, "pesq": 0.0001}
    urgent_weights = {"se": 5.0, "ssl": 100.0, "gan": 0.0001, "fm": 0.001, "pesq": 0.0001}

    def weighted(weights: dict[str, float]) -> float:
        return sum(float(raw[key]) * weight for key, weight in weights.items())

    urgent_scp = args.repo / "data/scp/scp_urgent/tr_s.scp"
    vctk_scp = args.repo / "data/scp/scp_VCTK/tr_s.scp"
    urgent_cfg = args.repo / "tf_restormer/models/TF_Restormer/configs/.legacy/URGENT.yaml"
    history_pattern = re.compile(r"^: (\d+):\d+;(.*)$")
    history_rows = []
    for line in args.history.read_text(errors="replace").splitlines():
        match = history_pattern.match(line)
        if not match:
            continue
        timestamp = int(match.group(1))
        command = match.group(2)
        if 1765335000 <= timestamp <= 1765348000 and "TF_Restormer" in command:
            history_rows.append({
                "timestamp": timestamp,
                "time_kst": datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Seoul")).isoformat(),
                "command": command,
            })
    baseline_train = [
        row for row in history_rows
        if "--engine_mode train --config baseline.yaml" in row["command"]
    ]
    urgent_eval = [
        row for row in history_rows
        if "--engine_mode eval --config URGENT.yaml" in row["command"]
    ]
    post_resume_intervals = [interval(epoch, epoch + 1) for epoch in range(20, 25)]
    result = {
        "schema_version": 1,
        "intervals": [
            interval(16, 19),
            interval(19, 20),
            *post_resume_intervals,
            interval(20, 25),
        ],
        "checkpoint_lr": {
            str(epoch): float(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"])
            for epoch, checkpoint in generator.items()
        },
        "epoch20_loss_reconstruction": {
            "checkpoint_train_loss": float(generator[20]["train_loss"]),
            "baseline_weights": baseline_weights,
            "baseline_total": weighted(baseline_weights),
            "baseline_error": weighted(baseline_weights) - float(generator[20]["train_loss"]),
            "urgent_snapshot_weights": urgent_weights,
            "urgent_snapshot_total": weighted(urgent_weights),
            "urgent_snapshot_error": weighted(urgent_weights) - float(generator[20]["train_loss"]),
        },
        "contemporaneous_files": {
            "urgent_train_scp_lines": len(urgent_scp.read_text().splitlines()),
            "vctk_train_scp_lines": len(vctk_scp.read_text().splitlines()),
            "urgent_config_mtime_ns": urgent_cfg.stat().st_mtime_ns,
            "urgent_scp_mtime_ns": urgent_scp.stat().st_mtime_ns,
            "urgent_first_clean_path": urgent_scp.read_text().splitlines()[0].split()[-1],
            "urgent_audio_present": Path(urgent_scp.read_text().splitlines()[0].split()[-1]).exists(),
        },
        "shell_history": {
            "source": str(args.history),
            "checkpoint20_mtime_kst": datetime.fromtimestamp(
                (log / "weights/epoch.0020.pth").stat().st_mtime,
                ZoneInfo("Asia/Seoul"),
            ).isoformat(),
            "relevant_commands": history_rows,
            "baseline_train_commands": baseline_train,
            "urgent_eval_commands": urgent_eval,
        },
        "inference": {
            "variable_rate_path_active_before_epoch20": abs(interval(16, 19)["padding_path_fraction"] - 0.85) < 0.01,
            "epoch20_generator_steps_match_urgent_scp": interval(19, 20)["generator_steps"] == len(urgent_scp.read_text().splitlines()),
            "epoch20_discriminator_two_updates_per_batch": interval(19, 20)["discriminator_optimizer_steps"] == 2 * interval(19, 20)["generator_steps"],
            "epoch20_uses_baseline_loss_weights": abs(weighted(baseline_weights) - float(generator[20]["train_loss"])) < 1e-8,
            "epoch20_does_not_use_preserved_urgent_loss_weights": abs(weighted(urgent_weights) - float(generator[20]["train_loss"])) > 1e-8,
            "resume_command_uses_baseline_config": len(baseline_train) == 1,
            "urgent_eval_preceded_baseline_resume": bool(
                baseline_train and urgent_eval and urgent_eval[-1]["timestamp"] < baseline_train[0]["timestamp"]
            ),
            "post_resume_epochs_are_1000_steps": all(
                row["generator_steps"] == 1000 for row in post_resume_intervals
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
