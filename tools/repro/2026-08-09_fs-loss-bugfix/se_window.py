"""se loss 윈도우가 fs_target에 따라 어떻게 달라지는지 실측."""
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

from tf_restormer.models.TF_Restormer.loss import MS_STFT_Gen_SC_Loss  # noqa: E402

FRAME_MS, HOP_MS = 40, 20   # config stft.frame_length / frame_shift
WIN_CFG = 1920              # config engine.*.loss_enhance.window_size

print("=== 고정 window_size=[1920] (hop=win//4=480) 의 물리 길이 ===")
print(f"{'fs_target':>10s} {'window':>10s} {'hop':>9s} {'FFT bin 간격':>12s} {'학습 비중':>9s}")
share = {48000: "40% (기본)", 44100: "20%", 24000: "20%", 16000: "20%"}
for fs in [48000, 44100, 24000, 16000]:
    print(f"{fs:>10d} {WIN_CFG/fs*1000:>8.1f}ms {WIN_CFG/4/fs*1000:>7.1f}ms {fs/WIN_CFG:>10.2f}Hz {share[fs]:>12s}")

print(f"\n(참고) 모델 STFT와 discriminator는 ms 기준으로 fs마다 재계산됨")
print(f"{'fs':>10s} {'모델 STFT window':>18s} {'se loss window':>16s}")
for fs in [48000, 44100, 24000, 16000]:
    model_win = int(FRAME_MS * fs / 1000)
    print(f"{fs:>10d} {model_win:>13d} samples {WIN_CFG:>11d} samples"
          f"   {'일치' if model_win == WIN_CFG else '불일치 (%.1fx)' % (WIN_CFG / model_win)}")

# ---- 실제 loss 값 차이 ----
print("\n=== 같은 신호 쌍에 대한 se loss 값 (고정 1920 vs ms 정렬) ===")
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clean, sr0 = sf.read("demo/samples/vctk_clean/16k_to_48k/clean/p360_072_mic1.wav", dtype="float32")
est, _ = sf.read("demo/samples/vctk_clean/16k_to_48k/our_TF_Restormer/p360_072_mic1.wav", dtype="float32")
if clean.ndim > 1:
    clean = clean[:, 0]
if est.ndim > 1:
    est = est[:, 0]
L = min(len(clean), len(est))
c48 = torch.from_numpy(clean[:L]).unsqueeze(0).to(dev)
e48 = torch.from_numpy(est[:L]).unsqueeze(0).to(dev)

print(f"{'fs_target':>10s} {'고정 1920':>12s} {'ms정렬':>12s} {'비율':>8s} {'프레임 수(고정/ms)':>20s}")
for fs in [48000, 44100, 24000, 16000]:
    c = AF.resample(c48, 48000, fs) if fs != 48000 else c48
    e = AF.resample(e48, 48000, fs) if fs != 48000 else e48
    win_ms = int(FRAME_MS * fs / 1000)
    l_fixed = MS_STFT_Gen_SC_Loss(window_size=[WIN_CFG], tau=1e-4, device=dev)
    l_msal = MS_STFT_Gen_SC_Loss(window_size=[win_ms], tau=1e-4, device=dev)
    with torch.no_grad():
        v_fixed = l_fixed(e, c).item()
        v_msal = l_msal(e, c).item()
    n_fixed = c.shape[-1] // (WIN_CFG // 4)
    n_ms = c.shape[-1] // (win_ms // 4)
    print(f"{fs:>10d} {v_fixed:>12.6f} {v_msal:>12.6f} {v_fixed/max(v_msal,1e-12):>7.2f}x "
          f"{n_fixed:>9d} / {n_ms:<8d}")
