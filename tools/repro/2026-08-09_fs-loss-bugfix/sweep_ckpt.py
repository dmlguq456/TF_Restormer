"""여러 학습 체크포인트에서 8kHz 이상 대역 생성 능력을 비교."""
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from tf_restormer import _config
from tf_restormer.models.TF_Restormer.model import Model
from tf_restormer.utils import util_stft

LOG = "tf_restormer/models/TF_Restormer/log"
CKPTS = [
    ("released model.pt", "tf_restormer/checkpoints/baseline/model.pt"),
    ("adv epoch.0025", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0025.pth"),
    ("adv epoch.0024", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0024.pth"),
    ("adv epoch.0023", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0023.pth"),
    ("adv epoch.0022", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0022.pth"),
    ("adv epoch.0021", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0021.pth"),
    ("adv epoch.0020 (12월)", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0020.pth"),
    ("adv epoch.0019 (9월)", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0019.pth"),
    ("adv epoch.0016 (9월)", f"{LOG}/log_adversarial_to48k_baseline.yaml/weights/epoch.0016.pth"),
    ("pretrain epoch.0015 (11월)", f"{LOG}/log_pretrain_to48k_baseline.yaml/weights/epoch.0015.pth"),
    ("pretrain epoch.0010 (12월)", f"{LOG}/log_pretrain_to48k_baseline.yaml/weights/epoch.0010.pth"),
]

cfg = _config.load_config("TF_Restormer", "baseline.yaml")["config"]
dev = torch.device("cuda:0")
stft16 = util_stft.STFT(640, 320, device=dev, normalize=True)

clean, sr0 = sf.read("data/valid_sample/UNIVERSE_sample/0015.wav", dtype="float32")
if clean.ndim > 1:
    clean = clean[:, 0]
x16 = torch.from_numpy(clean) if sr0 == 16000 else AF.resample(torch.from_numpy(clean), sr0, 16000)
X = stft16(x16.unsqueeze(0).to(dev), cplx=True)
xin = torch.stack([X.real, X.imag], dim=-1)
print(f"input F={X.shape[1]}, T={X.shape[2]}\n")
print(f"{'checkpoint':30s} {'8-12k':>8s} {'12-16k':>8s} {'16-20k':>8s} {'20-24k':>8s}   (dB rel. to 0-8k)")

for name, path in CKPTS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    sd = {k: v for k, v in sd.items() if v.dtype != torch.float64}
    model = Model(**cfg["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.band_detect_enable = False
    model.to(dev).eval()
    with torch.inference_mode():
        y = model(xin, out_F=961)
    mag = torch.sqrt(y[..., 0] ** 2 + y[..., 1] ** 2).mean(dim=(0, 2))
    ref = mag[:321].mean().item()
    cells = []
    for s in range(321, 961, 160):
        e = min(s + 160, 961)
        seg = mag[s:e].mean().item()
        cells.append(f"{20*np.log10((seg+1e-30)/(ref+1e-30)):+8.1f}")
    extra = f"  [missing={len(missing)}, unexpected={len(unexpected)}]" if (missing or unexpected) else ""
    print(f"{name:30s} " + " ".join(cells) + extra)
    del model
    torch.cuda.empty_cache()
