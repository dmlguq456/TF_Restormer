"""간이 학습 재현: downsample_src 도입과 손실 버그가 고주파에 미치는 영향.

epoch.0019(고주파 정상)에서 출발해 세 조건으로 짧게 학습하며 고주파 대역을 추적한다.

  A. pre-refactor   : downsample_src 없음 (타겟 항상 48k) — epoch<=19 방식
  B. buggy          : downsample_src 0.6 + se window 1920 샘플 고정 — epoch 20~25 방식
  C. fixed          : downsample_src 0.6 + se window 40ms 정렬 — 수정 후

한계: VCTK DB가 없어 repo 내 48kHz 샘플만 사용하고, 의존성이 없는 se loss만 쓴다
(실제 학습의 ssl/gan/fm/pesq 제외). downsample_src와 se window의 효과만 격리 관찰한다.
"""
import glob
import random
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def _stub(name, attrs):
    m = sys.modules.get(name) or type(sys)(name)
    sys.modules[name] = m
    for a in attrs:
        if not hasattr(m, a):
            setattr(m, a, type(a, (), {"from_pretrained": staticmethod(lambda *x, **k: None)}))


_stub("transformers", ("Wav2Vec2Model", "Wav2Vec2Processor", "WhisperModel"))
_stub("torch_pesq", ("PesqLoss",))

from tf_restormer import _config                                    # noqa: E402
from tf_restormer.models.TF_Restormer.model import Model            # noqa: E402
from tf_restormer.models.TF_Restormer.loss import MS_STFT_Gen_SC_Loss  # noqa: E402
from tf_restormer.utils import util_stft                            # noqa: E402

CKPT = "tf_restormer/models/TF_Restormer/log/log_adversarial_to48k_baseline.yaml/weights/epoch.0019.pth"
STEPS = 300
EVAL_EVERY = 50
LR = 1.6e-6          # epoch 20 기록값
FS_LIST_SRC = [16000, 24000, 44100]
PROB_DOWNSAMPLE_SRC = 0.6
DUR = 2.0
dev = torch.device("cuda:1")

cfg = _config.load_config("TF_Restormer", "baseline.yaml")["config"]
FRAME_MS, HOP_MS = cfg["stft"]["frame_length"], cfg["stft"]["frame_shift"]


def stft_for(fs):
    return util_stft.STFT(int(FRAME_MS * fs / 1000), int(HOP_MS * fs / 1000), device=dev, normalize=True)


def istft_for(fs):
    return util_stft.iSTFT(int(FRAME_MS * fs / 1000), int(HOP_MS * fs / 1000), device=dev, normalize=True)


STFT_C = {fs: stft_for(fs) for fs in [16000, 24000, 44100, 48000]}
ISTFT_C = {fs: istft_for(fs) for fs in [16000, 24000, 44100, 48000]}


# ---------------- data ----------------
def load_48k():
    out = []
    for f in sorted(glob.glob("demo/samples/**/clean/*.wav", recursive=True)):
        d, sr = sf.read(f, dtype="float32")
        if sr != 48000:
            continue
        if d.ndim > 1:
            d = d[:, 0]
        if len(d) >= int(DUR * 48000):
            out.append(d)
    return out


POOL = load_48k()
print(f"48kHz clean pool: {len(POOL)} files")

EVAL_SRC = POOL[0][: int(DUR * 48000)].copy()


def batch(rng, bs=1):
    xs = []
    for _ in range(bs):
        d = POOL[rng.randrange(len(POOL))]
        n = int(DUR * 48000)
        i = rng.randrange(0, max(1, len(d) - n))
        xs.append(d[i:i + n])
    return torch.from_numpy(np.stack(xs)).to(dev)


def make_input(clean48, rng):
    """clean 48k -> 16k 대역제한 noisy 입력 (모델 입력은 항상 16kHz 도메인)."""
    x16 = AF.resample(clean48, 48000, 16000)
    snr = rng.uniform(5, 20)
    n = torch.randn_like(x16)
    p_s, p_n = x16.pow(2).mean(), n.pow(2).mean()
    n = n * torch.sqrt(p_s / (p_n * 10 ** (snr / 10)))
    return x16 + n


def hf_ratio(wav48):
    v = wav48.detach().float().cpu().numpy().reshape(-1)
    n, hop = 2048, 1024
    w = np.hanning(n)
    fr = [v[i:i + n] * w for i in range(0, max(1, len(v) - n), hop)]
    P = (np.abs(np.fft.rfft(np.array(fr), axis=-1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / 48000)
    lo = P[(f >= 300) & (f < 3400)].mean()
    hi = P[(f >= 8000) & (f < 20000)].mean()
    return 10 * np.log10((hi + 1e-30) / (lo + 1e-30))


@torch.no_grad()
def evaluate(model):
    model.eval()
    x = torch.from_numpy(EVAL_SRC).unsqueeze(0).to(dev)
    x16 = AF.resample(x, 48000, 16000)
    X = STFT_C[16000](x16, cplx=True)
    y = model(torch.stack([X.real, X.imag], dim=-1), out_F=961)
    w = ISTFT_C[48000](torch.complex(y[..., 0], y[..., 1]), cplx=True, squeeze=False)
    model.train()
    return hf_ratio(w)


def run(tag, use_downsample_src, se_window_kwargs):
    rng = random.Random(1234)
    torch.manual_seed(1234)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model_state_dict"].items() if v.dtype != torch.float64}
    model = Model(**cfg["model"])
    model.load_state_dict(sd, strict=False)
    model.band_detect_enable = False
    model.to(dev).train()

    se = MS_STFT_Gen_SC_Loss(tau=1e-4, device=dev, **se_window_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.995), weight_decay=1e-2)

    traj = [(0, evaluate(model))]
    fs_hist = {}
    for step in range(1, STEPS + 1):
        clean48 = batch(rng)
        noisy16 = make_input(clean48, rng)

        if use_downsample_src and rng.random() < PROB_DOWNSAMPLE_SRC:
            fs_t = rng.choice(FS_LIST_SRC)
            tgt = AF.resample(clean48, 48000, fs_t)
        else:
            fs_t = 48000
            tgt = clean48
        fs_hist[fs_t] = fs_hist.get(fs_t, 0) + 1
        out_F = int(FRAME_MS * fs_t / 1000) // 2 + 1

        tgt_stft = STFT_C[fs_t](tgt, cplx=True)
        src_wav = ISTFT_C[fs_t](tgt_stft, cplx=True, squeeze=False)

        X = STFT_C[16000](noisy16, cplx=True)
        y = model(torch.stack([X.real, X.imag], dim=-1), out_F=out_F)
        out_wav = ISTFT_C[fs_t](torch.complex(y[..., 0], y[..., 1]), cplx=True, squeeze=False)

        L = min(out_wav.shape[-1], src_wav.shape[-1])
        loss = se(out_wav[..., :L], src_wav[..., :L], epoch=20, fs=fs_t)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()

        if step % EVAL_EVERY == 0:
            traj.append((step, evaluate(model)))

    print(f"\n[{tag}]  target fs 분포: " + ", ".join(f"{k}:{v}" for k, v in sorted(fs_hist.items())))
    print("   step " + "".join(f"{s:>9d}" for s, _ in traj))
    print("   HF/LF" + "".join(f"{v:>8.1f}dB" for _, v in traj))
    del model
    torch.cuda.empty_cache()
    return traj


print(f"\nepoch.0019 에서 출발, {STEPS} step, lr={LR:g}, se loss only\n")
a = run("A pre-refactor (downsample_src 없음, 48k 타겟 고정)", False, {"window_size": [1920]})
b = run("B buggy       (downsample_src 0.6 + window 1920 고정)", True, {"window_size": [1920]})
c = run("C fixed       (downsample_src 0.6 + window 40ms 정렬)", True, {"window_ms": [40]})

print("\n=== 요약: 고주파(8-20kHz) 비율 변화 ===")
print(f"{'조건':<46s} {'시작':>9s} {'끝':>9s} {'변화':>9s}")
for tag, t in [("A pre-refactor", a), ("B buggy (refactor 방식)", b), ("C fixed", c)]:
    print(f"{tag:<46s} {t[0][1]:>7.1f}dB {t[-1][1]:>7.1f}dB {t[-1][1]-t[0][1]:>+7.1f}dB")
