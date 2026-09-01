"""se loss window ms 정렬 검증."""
import sys
import torch


def _stub(name, attrs):
    m = sys.modules.get(name) or type(sys)(name)
    sys.modules[name] = m
    for a in attrs:
        if not hasattr(m, a):
            setattr(m, a, type(a, (), {"from_pretrained": staticmethod(lambda *x, **k: None)}))


_stub("transformers", ("Wav2Vec2Model", "Wav2Vec2Processor", "WhisperModel"))
_stub("torch_pesq", ("PesqLoss",))

from tf_restormer.models.TF_Restormer.loss import MS_STFT_Gen_SC_Loss  # noqa: E402

dev = "cpu"
ok = True

print("=== 수정 후: window_ms=[40] — fs별 실제 윈도우 ===")
print(f"{'fs':>8s} {'window(samples)':>16s} {'= ms':>8s} {'hop':>8s} {'판정':>6s}")
L = MS_STFT_Gen_SC_Loss(window_ms=[40], tau=1e-4, device=dev)
for fs in [48000, 44100, 32000, 24000, 22050, 16000, 8000]:
    st = L._get_stfts(fs)[0]
    win = st.N
    ms = win * 1000.0 / fs
    good = abs(ms - 40.0) < 0.1
    ok &= good
    print(f"{fs:>8d} {win:>16d} {ms:>7.1f}ms {st.stride:>8d} {'OK' if good else 'FAIL':>6s}")

print("\n=== 하위호환: 구 config window_size=[1920] (48kHz 샘플) ===")
Lold = MS_STFT_Gen_SC_Loss(window_size=[1920], tau=1e-4, device=dev)
print(f"  해석된 window_ms = {Lold.window_ms}  (기대 [40.0])")
ok &= Lold.window_ms == [40.0]
for fs in [48000, 16000]:
    st = Lold._get_stfts(fs)[0]
    win = st.N
    print(f"  fs={fs}: {win} samples = {win*1000/fs:.1f}ms")

print("\n=== 회귀: 48kHz 에서 수정 전과 동일한 값인가 ===")
torch.manual_seed(0)
a = torch.randn(1, 48000 * 2) * 0.1
b = a + torch.randn(1, 48000 * 2) * 0.01
new48 = MS_STFT_Gen_SC_Loss(window_ms=[40], tau=1e-4, device=dev)(b, a, fs=48000).item()
legacy = MS_STFT_Gen_SC_Loss(window_size=[1920], tau=1e-4, device=dev)(b, a, fs=48000).item()
nofs = MS_STFT_Gen_SC_Loss(window_ms=[40], tau=1e-4, device=dev)(b, a).item()
print(f"  window_ms=[40], fs=48000 : {new48:.8f}")
print(f"  window_size=[1920] 하위호환: {legacy:.8f}")
print(f"  fs 생략 (48k 기본값)      : {nofs:.8f}")
same = abs(new48 - legacy) < 1e-12 and abs(new48 - nofs) < 1e-12
ok &= same
print(f"  세 경로 일치: {same}")

print("\n=== 수정 전/후: 16kHz 타겟에서 실제로 쓰이는 윈도우 ===")
print(f"  수정 전(고정 1920): {1920} samples = {1920*1000/16000:.0f}ms")
st16 = L._get_stfts(16000)[0]
print(f"  수정 후(ms 정렬)  : {st16.N} samples = {st16.N*1000/16000:.0f}ms")

print("\n" + ("ALL PASS" if ok else "FAILED"))
sys.exit(0 if ok else 1)
