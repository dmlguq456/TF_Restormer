"""Verify an exported epoch-19 directory by state, config, and demo inference."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import yaml

from tf_restormer import SEInference
from tf_restormer.inference import _strip_profiling_keys
from tf_restormer.utils.util_engine import _fix_compiled_state_dict


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_ratio(waveform: np.ndarray, sample_rate: int = 48000) -> float:
    n_fft, hop = 2048, 1024
    window = np.hanning(n_fft)
    frames = [
        waveform[index:index + n_fft] * window
        for index in range(0, max(1, len(waveform) - n_fft), hop)
    ]
    power = (np.abs(np.fft.rfft(np.asarray(frames), axis=-1)) ** 2).mean(axis=0)
    frequency = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    low = power[(frequency >= 300) & (frequency < 3400)].mean()
    high = power[(frequency >= 8000) & (frequency < 20000)].mean()
    return float(10 * np.log10((high + 1e-30) / (low + 1e-30)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=0.75)
    args = parser.parse_args()

    model_path = args.artifact_dir / "model.pt"
    config_path = args.artifact_dir / "config.yaml"
    metadata_path = args.artifact_dir / "export_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["config"]

    raw = torch.load(args.raw_checkpoint, map_location="cpu", weights_only=True)
    raw_state = _strip_profiling_keys(
        _fix_compiled_state_dict(raw["model_state_dict"])
    )
    exported = torch.load(model_path, map_location="cpu", weights_only=True)
    exported_state = exported["model_state_dict"]
    same_keys = set(raw_state) == set(exported_state)
    unequal = [
        key for key in sorted(set(raw_state) & set(exported_state))
        if not torch.equal(raw_state[key], exported_state[key])
    ]

    inference = SEInference.from_pretrained(
        checkpoint_path=args.artifact_dir,
        device=args.device,
    )
    band_detect_runtime = bool(inference.engine.model.band_detect_enable)
    demo = args.repo / "demo/samples/vctk_clean/16k_to_48k/input"
    files = [Path(path) for path in sorted(glob.glob(str(demo / "*.wav")))[:args.samples]]
    rows = []
    for path in files:
        waveform, sample_rate = sf.read(path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        waveform = waveform[: int(args.seconds * sample_rate)]
        native16 = AF.resample(torch.from_numpy(waveform), sample_rate, 16000)
        with torch.inference_mode():
            result = inference.process_waveform(
                native16.unsqueeze(0),
                fs_in=16000,
                fs_out=48000,
                mode="single_pass",
            )
        output = result["waveform"].squeeze(0).detach().cpu().float().numpy()
        rows.append({
            "sample": path.name,
            "hf_lf_db": hf_ratio(output),
            "waveform_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "samples": int(output.size),
        })

    mean_hf = float(np.mean([row["hf_lf_db"] for row in rows]))
    checks = {
        "metadata_epoch_19": metadata.get("source_epoch") == 19,
        "model_hash_matches_metadata": file_hash(model_path) == metadata.get("model_sha256"),
        "config_hash_matches_metadata": file_hash(config_path) == metadata.get("config_sha256"),
        "raw_state_keys_exact": same_keys,
        "raw_state_tensors_exact": same_keys and not unequal,
        "config_band_detect_enabled": bool(config["model"]["band_detect"]["enable"]),
        "runtime_band_detect_enabled": band_detect_runtime,
        "three_demo_samples": len(rows) == args.samples,
        "healthy_high_frequency": mean_hf > -45.0,
    }
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        checks["reference_waveforms_bit_exact"] = [
            row["waveform_sha256"] for row in rows
        ] == [row["waveform_sha256"] for row in reference["demo"]]
        checks["reference_hf_exact"] = abs(mean_hf - reference["mean_hf_lf_db"]) < 1e-12

    result = {
        "schema_version": 1,
        "artifact_dir": str(args.artifact_dir),
        "model_sha256": file_hash(model_path),
        "config_sha256": file_hash(config_path),
        "source_checkpoint_sha256": file_hash(args.raw_checkpoint),
        "source_epoch": metadata.get("source_epoch"),
        "state_key_count": len(exported_state),
        "unequal_state_keys": unequal,
        "band_detect_runtime": band_detect_runtime,
        "mean_hf_lf_db": mean_hf,
        "demo": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
