"""Quantify the two fs-loss bugs on epoch-19 outputs using a tiny fixed set."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from tf_restormer import _config
from tf_restormer.models.TF_Restormer.loss import MS_STFT_Gen_SC_Loss, SSL_FM_Loss
from tf_restormer.models.TF_Restormer.model import Model
from tf_restormer.utils import util_stft


RATES = (16000, 24000, 44100, 48000)
RATE_WEIGHTS = {16000: 0.2, 24000: 0.2, 44100: 0.2, 48000: 0.4}
OBSERVED_GAP = 0.00976813844932496 - 0.0036484815806017454


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mono(path: str, seconds: float) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    n = min(len(data), int(sr * seconds))
    return torch.from_numpy(data[:n]).unsqueeze(0), sr


def mean_by_rate(rows: list[dict], key: str) -> dict[int, float]:
    return {
        fs: float(np.mean([row[key] for row in rows if row["fs"] == fs]))
        for fs in RATES
    }


def weighted(values: dict[int, float]) -> float:
    return sum(RATE_WEIGHTS[fs] * values[fs] for fs in RATES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=1.5)
    args = parser.parse_args()

    torch.manual_seed(190020)
    device = torch.device(args.device)
    cfg = _config.load_config("TF_Restormer", "baseline.yaml")["config"]
    frame_ms = cfg["stft"]["frame_length"]
    hop_ms = cfg["stft"]["frame_shift"]

    stft = {
        fs: util_stft.STFT(
            int(frame_ms * fs / 1000), int(hop_ms * fs / 1000),
            device=device, normalize=True,
        )
        for fs in RATES
    }
    istft = {
        fs: util_stft.iSTFT(
            int(frame_ms * fs / 1000), int(hop_ms * fs / 1000),
            device=device, normalize=True,
        )
        for fs in RATES
    }

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if value.dtype != torch.float64
    }
    model = Model(**cfg["model"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.band_detect_enable = False
    model.to(device).eval()

    loss_cfg = cfg["engine"][cfg["train_phase"]]
    ssl = SSL_FM_Loss(**loss_cfg["loss_rep"], device=device)
    se_fixed = MS_STFT_Gen_SC_Loss(window_ms=[40], tau=1e-4, device=device)
    # Passing fs=48k on a lower-rate tensor reproduces the old fixed 1,920-sample
    # analysis window without duplicating the loss implementation.
    se_buggy = MS_STFT_Gen_SC_Loss(window_ms=[40], tau=1e-4, device=device)

    demo = args.repo / "demo/samples/vctk_clean/16k_to_48k"
    input_files = sorted(glob.glob(str(demo / "input/*.wav")))[: args.samples]
    if len(input_files) != args.samples:
        raise RuntimeError(f"requested {args.samples} samples, found {len(input_files)}")

    rows: list[dict] = []
    with torch.inference_mode():
        for input_path in input_files:
            clean_path = str(demo / "clean" / Path(input_path).name)
            input_48, input_sr = load_mono(input_path, args.seconds)
            clean_48, clean_sr = load_mono(clean_path, args.seconds)
            if input_sr != 48000 or clean_sr != 48000:
                raise RuntimeError("demo pair is not 48 kHz")
            input_16 = AF.resample(input_48, 48000, 16000).to(device)
            input_spec = stft[16000](input_16, cplx=True)
            model_input = torch.stack([input_spec.real, input_spec.imag], dim=-1)

            for fs in RATES:
                target = clean_48.to(device) if fs == 48000 else AF.resample(clean_48, 48000, fs).to(device)
                target_spec = stft[fs](target, cplx=True)
                src_wav = istft[fs](target_spec, cplx=True, squeeze=False)
                out_f = int(frame_ms * fs / 1000) // 2 + 1
                comp = model(model_input, out_F=out_f)
                out_spec = torch.complex(comp[..., 0], comp[..., 1])
                out_wav = istft[fs](out_spec, cplx=True, squeeze=False)
                length = min(out_wav.shape[-1], src_wav.shape[-1])
                out_wav, src_wav = out_wav[..., :length], src_wav[..., :length]

                se_fix = float(se_fixed(out_wav, src_wav, epoch=20, fs=fs))
                se_bug = float(se_buggy(out_wav, src_wav, epoch=20, fs=48000))
                ssl_fix = float(ssl(out_wav, src_wav, fs=fs))
                ssl_bug = float(ssl(out_wav, src_wav))
                rows.append({
                    "sample": Path(input_path).name,
                    "fs": fs,
                    "samples": length,
                    "se_fixed": se_fix,
                    "se_buggy": se_bug,
                    "ssl_fixed": ssl_fix,
                    "ssl_buggy": ssl_bug,
                    "affected_fixed": se_fix + 100.0 * ssl_fix,
                    "affected_buggy": se_bug + 100.0 * ssl_bug,
                })
                print(
                    f"{Path(input_path).name} fs={fs}: "
                    f"se {se_bug:.6g}->{se_fix:.6g}, "
                    f"ssl*100 {100*ssl_bug:.6g}->{100*ssl_fix:.6g}"
                )

    summaries = {
        key: mean_by_rate(rows, key)
        for key in ("se_fixed", "se_buggy", "ssl_fixed", "ssl_buggy", "affected_fixed", "affected_buggy")
    }
    expected_fixed = weighted(summaries["affected_fixed"])
    expected_buggy = weighted(summaries["affected_buggy"])
    delta = expected_buggy - expected_fixed
    explained_fraction = delta / OBSERVED_GAP
    ref = summaries["affected_fixed"][48000]
    fixed_dev = abs(expected_fixed - ref)
    buggy_dev = abs(expected_buggy - ref)
    closeness_improvement = buggy_dev / max(fixed_dev, 1e-12)
    primary_cause = bool(
        delta > 0
        and explained_fraction >= 0.5
        and closeness_improvement >= 2.0
    )

    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_train_loss": float(checkpoint["train_loss"]),
        "checkpoint_valid_loss": float(checkpoint["valid_loss"]),
        "samples": input_files,
        "duration_seconds": args.seconds,
        "rate_weights": RATE_WEIGHTS,
        "observed_epoch19_to_20_gap": OBSERVED_GAP,
        "model_load": {"missing": list(missing), "unexpected": list(unexpected)},
        "rows": rows,
        "mean_by_rate": summaries,
        "decision": {
            "expected_affected_fixed": expected_fixed,
            "expected_affected_buggy": expected_buggy,
            "bug_delta": delta,
            "explained_fraction": explained_fraction,
            "fixed_deviation_from_48k": fixed_dev,
            "buggy_deviation_from_48k": buggy_dev,
            "closeness_improvement": closeness_improvement,
            "primary_cause": primary_cause,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
