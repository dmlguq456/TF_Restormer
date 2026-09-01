"""논문 데모(16k->48k) 출력과 각 epoch 체크포인트 출력을 대조해 논문 버전을 특정."""
import glob
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from tf_restormer import _config
from tf_restormer.models.TF_Restormer.model import Model
from tf_restormer.utils import util_stft

D = "demo/samples/vctk_clean/16k_to_48k"
LOG = "tf_restormer/models/TF_Restormer/log"
CKPTS = [
    ("adv epoch.0016", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0016.pth"),
    ("adv epoch.0019", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0019.pth"),
    ("adv epoch.0020", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0020.pth"),
    ("adv epoch.0025 (배포)", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0025.pth"),
    ("pretrain epoch.0015", f"{LOG}/log_pretrain_to48k_baseline.yaml/weights/epoch.0015.pth"),
]

dev = torch.device("cuda:0")
cfg = _config.load_config("TF_Restormer", "baseline.yaml")["config"]
stft16 = util_stft.STFT(640, 320, device=dev, normalize=True)
istft48 = util_stft.iSTFT(1920, 960, device=dev, normalize=True)

files = sorted(glob.glob(f"{D}/input/*.wav"))[:5]
print(f"{len(files)} demo files\n")


def logspec(x, n=1024):
    hop = n // 4
    w = np.hanning(n)
    fr = [x[i:i + n] * w for i in range(0, max(1, len(x) - n), hop)]
    S = np.abs(np.fft.rfft(np.array(fr), axis=-1))
    return np.log10(S + 1e-8)


def lsd(a, b, sr=48000):
    L = min(len(a), len(b))
    A, B = logspec(a[:L]), logspec(b[:L])
    T = min(len(A), len(B))
    return float(np.sqrt(((A[:T] - B[:T]) ** 2).mean()))


def hf_ratio(x, sr=48000):
    """8-20kHz 대비 0.3-3.4kHz 에너지 (dB)"""
    n = 2048
    hop = n // 2
    w = np.hanning(n)
    fr = [x[i:i + n] * w for i in range(0, max(1, len(x) - n), hop)]
    P = (np.abs(np.fft.rfft(np.array(fr), axis=-1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    lo = P[(f >= 300) & (f < 3400)].mean()
    hi = P[(f >= 8000) & (f < 20000)].mean()
    return 10 * np.log10((hi + 1e-30) / (lo + 1e-30))


# 참조: 데모 출력과 clean의 특성
print("=== 데모 페이지 자체 (참조) ===")
for tag, sub in [("input (16k 대역)", "input"), ("demo our_TF_Restormer", "our_TF_Restormer"), ("clean (48k 정답)", "clean")]:
    vals, ls = [], []
    for fp in files:
        name = fp.split("/")[-1]
        d, sr = sf.read(f"{D}/{sub}/{name}", dtype="float32")
        if d.ndim > 1:
            d = d[:, 0]
        vals.append(hf_ratio(d, sr))
        c, _ = sf.read(f"{D}/clean/{name}", dtype="float32")
        if c.ndim > 1:
            c = c[:, 0]
        ls.append(lsd(d, c))
    print(f"  {tag:24s} HF/LF={np.mean(vals):+6.1f} dB   LSD(vs clean)={np.mean(ls):.4f}")

print("\n=== 각 체크포인트로 데모 입력 재처리 ===")
print(f"{'checkpoint':24s} {'HF/LF':>8s} {'LSD vs demo':>12s} {'LSD vs clean':>13s}")
for name, path in CKPTS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model_state_dict"].items() if v.dtype != torch.float64}
    m = Model(**cfg["model"])
    m.load_state_dict(sd, strict=False)
    m.band_detect_enable = False
    m.to(dev).eval()

    hfs, l_demo, l_clean = [], [], []
    for fp in files:
        fn = fp.split("/")[-1]
        x, sr = sf.read(fp, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        # 데모 input은 48k 컨테이너의 16k 대역 신호 -> 네이티브 16k로 되돌려 입력
        x16 = AF.resample(torch.from_numpy(x), sr, 16000).to(dev)
        X = stft16(x16.unsqueeze(0), cplx=True)
        with torch.inference_mode():
            y = m(torch.stack([X.real, X.imag], dim=-1), out_F=961)
        out = istft48(torch.complex(y[..., 0], y[..., 1]), cplx=True, squeeze=False)
        o = out.squeeze(0).cpu().numpy()

        dem, _ = sf.read(f"{D}/our_TF_Restormer/{fn}", dtype="float32")
        cln, _ = sf.read(f"{D}/clean/{fn}", dtype="float32")
        if dem.ndim > 1:
            dem = dem[:, 0]
        if cln.ndim > 1:
            cln = cln[:, 0]
        hfs.append(hf_ratio(o))
        l_demo.append(lsd(o, dem))
        l_clean.append(lsd(o, cln))
    print(f"{name:24s} {np.mean(hfs):+8.1f} {np.mean(l_demo):>12.4f} {np.mean(l_clean):>13.4f}")
    del m
    torch.cuda.empty_cache()
