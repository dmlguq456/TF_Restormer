"""Compare the epoch-19→20 estimator update with full/band-limited targets."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

# Running this archived script by path otherwise lets the virtualenv's editable
# install of an older checkout win module resolution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from tf_restormer import _config
from tf_restormer.models.TF_Restormer.loss import MS_STFT_Gen_SC_Loss
from tf_restormer.models.TF_Restormer.model import Model
from tf_restormer.utils import util_stft


def state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if value.dtype != torch.float64
    }


def load_mono(path: Path, seconds: float) -> torch.Tensor:
    data, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 48000:
        raise RuntimeError(f"expected 48 kHz: {path} ({sample_rate})")
    if data.ndim > 1:
        data = data[:, 0]
    length = min(len(data), int(seconds * sample_rate))
    return torch.from_numpy(data[:length]).unsqueeze(0)


def hf_ratio(waveform: torch.Tensor) -> float:
    value = waveform.detach().float().cpu().numpy().reshape(-1)
    n_fft, hop = 2048, 1024
    window = np.hanning(n_fft)
    frames = [
        value[index:index + n_fft] * window
        for index in range(0, max(1, len(value) - n_fft), hop)
    ]
    power = (np.abs(np.fft.rfft(np.asarray(frames), axis=-1)) ** 2).mean(axis=0)
    freq = np.fft.rfftfreq(n_fft, 1 / 48000)
    low = power[(freq >= 300) & (freq < 3400)].mean()
    high = power[(freq >= 8000) & (freq < 20000)].mean()
    return float(10 * np.log10((high + 1e-30) / (low + 1e-30)))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left, right) / (left.norm() * right.norm() + 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=0.75)
    args = parser.parse_args()

    torch.manual_seed(1920)
    device = torch.device(args.device)
    cfg = _config.load_config("TF_Restormer", "baseline.yaml")["config"]
    log = args.repo / "tf_restormer/models/TF_Restormer/log/log_adversarial_to48k_baseline.yaml/weights"
    state19 = state(log / "epoch.0019.pth")
    state20 = state(log / "epoch.0020.pth")

    stft16 = util_stft.STFT(640, 320, device=device, normalize=True)
    istft48 = util_stft.iSTFT(1920, 960, device=device, normalize=True)
    loss_fn = MS_STFT_Gen_SC_Loss(window_ms=[40], tau=1e-4, device=device)
    demo = args.repo / "demo/samples/vctk_clean/16k_to_48k"
    inputs = [Path(path) for path in sorted(glob.glob(str(demo / "input/*.wav")))[:args.samples]]
    samples = []
    for input_path in inputs:
        clean = load_mono(demo / "clean" / input_path.name, args.seconds).to(device)
        input_wave = load_mono(input_path, args.seconds).to(device)
        bandlimited = AF.resample(AF.resample(clean, 48000, 16000), 16000, 48000)
        samples.append((input_path.name, input_wave, clean, bandlimited))

    def build(model_state: dict[str, torch.Tensor]) -> Model:
        model = Model(**cfg["model"])
        model.load_state_dict(model_state, strict=False)
        model.band_detect_enable = False
        model.to(device).eval()
        return model

    def forward(model: Model, input_wave: torch.Tensor) -> torch.Tensor:
        input16 = AF.resample(input_wave, 48000, 16000)
        spectrum = stft16(input16, cplx=True)
        output = model(torch.stack([spectrum.real, spectrum.imag], dim=-1), out_F=961)
        waveform = istft48(torch.complex(output[..., 0], output[..., 1]), cplx=True, squeeze=False)
        return waveform

    def mean_loss(model: Model, target_kind: str) -> float:
        values = []
        with torch.no_grad():
            for _, input_wave, full, band in samples:
                output = forward(model, input_wave)
                target = full if target_kind == "fullband" else band
                length = min(output.shape[-1], target.shape[-1])
                values.append(float(loss_fn(output[..., :length], target[..., :length], epoch=20, fs=48000)))
        return float(np.mean(values))

    model19 = build(state19)
    model20 = build(state20)
    estimator_names = [name for name, _ in model19.named_parameters() if name.startswith("estimator.")]
    delta = torch.cat([
        (state20[name] - state19[name]).reshape(-1).float()
        for name in estimator_names
    ]).to(device)

    def gradient(target_kind: str) -> torch.Tensor:
        for parameter in model19.parameters():
            parameter.requires_grad_(False)
        for parameter in model19.estimator.parameters():
            parameter.requires_grad_(True)
        model19.zero_grad(set_to_none=True)
        total = 0.0
        for _, input_wave, full, band in samples:
            output = forward(model19, input_wave)
            target = full if target_kind == "fullband" else band
            length = min(output.shape[-1], target.shape[-1])
            total = total + loss_fn(output[..., :length], target[..., :length], epoch=20, fs=48000)
        (total / len(samples)).backward()
        return torch.cat([
            dict(model19.named_parameters())[name].grad.reshape(-1).float()
            for name in estimator_names
        ]).detach().clone()

    grad_full = gradient("fullband")
    grad_band = gradient("bandlimited")

    def counterfactual(gradient_vector: torch.Tensor) -> Model:
        model = build(state19)
        direction = -gradient_vector
        direction = direction * (delta.norm() / (direction.norm() + 1e-30))
        offset = 0
        with torch.no_grad():
            parameters = dict(model.named_parameters())
            for name in estimator_names:
                parameter = parameters[name]
                count = parameter.numel()
                parameter.add_(direction[offset:offset + count].view_as(parameter))
                offset += count
        return model

    counter_full = counterfactual(grad_full)
    counter_band = counterfactual(grad_band)
    models = {
        "epoch19": model19,
        "epoch20": model20,
        "counterfactual_fullband": counter_full,
        "counterfactual_bandlimited": counter_band,
    }
    target_hf = {"fullband": [], "bandlimited": []}
    output_hf = defaultdict(list)
    with torch.no_grad():
        for _, input_wave, full, band in samples:
            target_hf["fullband"].append(hf_ratio(full))
            target_hf["bandlimited"].append(hf_ratio(band))
            for name, model in models.items():
                output_hf[name].append(hf_ratio(forward(model, input_wave)))

    losses = {
        model_name: {
            target: mean_loss(model, target)
            for target in ("fullband", "bandlimited")
        }
        for model_name, model in models.items()
    }
    result = {
        "schema_version": 1,
        "samples": [name for name, *_ in samples],
        "seconds": args.seconds,
        "estimator_parameter_count": int(delta.numel()),
        "actual_estimator_delta_l2": float(delta.norm()),
        "gradient_alignment": {
            "actual_delta_vs_fullband_descent": cosine(delta, -grad_full),
            "actual_delta_vs_bandlimited_descent": cosine(delta, -grad_band),
            "fullband_vs_bandlimited_descent": cosine(-grad_full, -grad_band),
        },
        "mean_hf_lf_db": {
            "targets": {key: float(np.mean(value)) for key, value in target_hf.items()},
            "outputs": {key: float(np.mean(value)) for key, value in output_hf.items()},
        },
        "spectral_loss": losses,
        "actual_update_effect": {
            "hf_lf_delta_db": float(np.mean(output_hf["epoch20"]) - np.mean(output_hf["epoch19"])),
            "fullband_loss_delta": losses["epoch20"]["fullband"] - losses["epoch19"]["fullband"],
            "bandlimited_loss_delta": losses["epoch20"]["bandlimited"] - losses["epoch19"]["bandlimited"],
        },
        "estimator_state_delta": {
            name: {
                "l2": float((state20[name] - state19[name]).float().norm()),
                "mean": float((state20[name] - state19[name]).float().mean()),
            }
            for name in estimator_names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
