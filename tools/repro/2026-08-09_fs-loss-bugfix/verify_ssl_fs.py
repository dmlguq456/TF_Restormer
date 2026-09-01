"""SSL_FM_Loss fs 처리 검증.

WavLM(transformers) 없이 리샘플 경로만 검증한다. feat_extractor는 항등 스텁으로 대체하고,
forward가 SSL 모델에 실제로 넘기는 신호를 가로채 샘플레이트/대역폭을 측정한다.
"""
import sys
import numpy as np
import torch

def _stub(mod_name, attrs):
    m = sys.modules.get(mod_name)
    if m is None:
        m = type(sys)(mod_name)
        sys.modules[mod_name] = m
    for a in attrs:
        if not hasattr(m, a):
            setattr(m, a, type(a, (), {"from_pretrained": staticmethod(lambda *x, **k: None)}))
    return m


_stub("transformers", ("Wav2Vec2Model", "Wav2Vec2Processor", "WhisperModel"))
_stub("torch_pesq", ("PesqLoss",))

from tf_restormer.models.TF_Restormer.loss import SSL_FM_Loss  # noqa: E402

TARGET_FS = 16000
SEEN = {}


def make_loss(device="cpu"):
    """WavLM 로드를 건너뛰고 리샘플 상태만 갖춘 인스턴스."""
    obj = SSL_FM_Loss.__new__(SSL_FM_Loss)
    torch.nn.Module.__init__(obj)
    obj.device = device
    obj.default_fs = 48000
    obj.target_fs = TARGET_FS
    obj.resamplers = {}
    obj.resamplers[obj.default_fs] = obj._build_resampler(obj.default_fs)

    def spy(x):
        SEEN["wav"] = x.detach().clone()
        return x.unsqueeze(1)

    obj.feat_extractor = spy
    return obj


def sweep(fs, dur=1.0):
    """0~Nyquist 선형 스윕 — 리샘플 후 남은 대역을 재면 실효 SR을 알 수 있다."""
    t = torch.arange(int(fs * dur)) / fs
    f1 = fs / 2 * 0.98
    return torch.sin(2 * np.pi * (f1 / (2 * dur)) * t ** 2).unsqueeze(0)


def eff_bandwidth(x, fs, floor_db=-40.0):
    """에너지가 피크 대비 floor_db 이상인 최고 주파수."""
    v = x.squeeze().numpy().astype(np.float64)
    n = 1 << int(np.ceil(np.log2(len(v))))
    P = np.abs(np.fft.rfft(v, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    # 100Hz 대역으로 평활
    k = max(1, int(100 / (f[1] - f[0])))
    Ps = np.convolve(P, np.ones(k) / k, mode="same")
    thr = Ps.max() * 10 ** (floor_db / 10)
    act = np.where(Ps > thr)[0]
    return float(f[act.max()]) if len(act) else 0.0


print("=== 수정 후: forward(out, src, fs) 가 SSL 모델에 넘기는 신호 ===")
print(f"{'입력 fs':>10s} {'입력 길이':>10s} {'전달 길이':>10s} {'기대 길이':>10s} {'전달 대역폭':>12s} {'판정':>6s}")
loss = make_loss()
ok_all = True
for fs in [48000, 44100, 24000, 16000]:
    x = sweep(fs)
    loss(x, x.clone(), fs)
    got = SEEN["wav"]
    expect_len = round(x.shape[-1] * TARGET_FS / fs)
    bw = eff_bandwidth(got, TARGET_FS)
    # 리샘플 후 신호는 min(원 Nyquist, 8k)까지 차 있어야 한다
    expect_bw = min(fs / 2, TARGET_FS / 2) * 0.95
    ok = abs(got.shape[-1] - expect_len) <= 2 and bw > expect_bw * 0.9
    ok_all &= ok
    print(f"{fs:>10d} {x.shape[-1]:>10d} {got.shape[-1]:>10d} {expect_len:>10d} {bw:>10.0f}Hz {'OK' if ok else 'FAIL':>6s}")

print("\n=== 수정 전 동작 재현 (항상 48k->16k 고정) ===")
print(f"{'입력 fs':>10s} {'전달 길이':>10s} {'전달 대역폭':>12s} {'실효 SR':>10s}")
fixed = loss.resamplers[48000]
for fs in [48000, 44100, 24000, 16000]:
    x = sweep(fs)
    got = fixed(loss.normalize(x))
    bw = eff_bandwidth(got, TARGET_FS)
    eff_sr = fs * TARGET_FS / 48000
    print(f"{fs:>10d} {got.shape[-1]:>10d} {bw:>10.0f}Hz {eff_sr:>9.0f}Hz")

print("\n=== 회귀: fs 생략 시 기존 동작(48k 가정) 유지 ===")
x = sweep(48000)
loss(x, x.clone())
a = SEEN["wav"].clone()
loss(x, x.clone(), 48000)
b = SEEN["wav"].clone()
same = torch.allclose(a, b, atol=0, rtol=0)
print(f"  forward(out,src) == forward(out,src,48000): {same}")
ok_all &= same

print("\n=== 리샘플러 캐시 ===")
print(f"  캐시된 fs: {sorted(loss.resamplers)}  (48000은 __init__, 나머지는 lazy)")

print("\n" + ("ALL PASS" if ok_all else "FAILED"))
sys.exit(0 if ok_all else 1)
