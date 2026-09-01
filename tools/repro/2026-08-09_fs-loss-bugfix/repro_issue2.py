"""Issue #2 재현: 16kHz 콘텐츠를 48kHz로 미리 업샘플해 넣으면 BWE가 죽는지 확인."""
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from tf_restormer import SEInference

CKPT = "tf_restormer/checkpoints/baseline"
SAMPLE = "data/valid_sample/UNIVERSE_sample/0015.wav"


def band_energy(wav, sr, lo, hi):
    x = wav.astype(np.float64)
    n = 1 << 15
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    f = np.abs(np.fft.rfft(x[:n])) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    m = (freqs >= lo) & (freqs < hi)
    return float(f[m].sum())


def report(tag, wav, sr):
    lo = band_energy(wav, sr, 0, 7800)
    hi = band_energy(wav, sr, 8200, 24000)
    ratio_db = 10 * np.log10((hi + 1e-20) / (lo + 1e-20))
    print(f"  [{tag}] sr={sr}  E(<7.8k)={lo:.3e}  E(8.2k~24k)={hi:.3e}  HF/LF = {ratio_db:+.1f} dB")


def main(enable_band_detect):
    model = SEInference.from_pretrained(checkpoint_path=CKPT, device="cuda:0")
    inner = model.engine.model
    print(f"config band_detect_enable = {inner.band_detect_enable}")
    if enable_band_detect:
        inner.band_detect_enable = True
        inner.band_db_below_peak = 60.0
        print(f"→ forced band_detect_enable = {inner.band_detect_enable}")

    clean, sr0 = sf.read(SAMPLE, dtype="float32")
    if clean.ndim > 1:
        clean = clean[:, 0]
    print(f"source: {SAMPLE} sr={sr0} len={len(clean)/sr0:.2f}s")

    t = torch.from_numpy(clean)
    # 16kHz 대역 제한 신호 (이슈 제출자의 입력과 동일한 상황)
    x16 = AF.resample(t, sr0, 16000)

    # (A) 네이티브 16kHz 입력
    out_a = model.process_waveform(x16.unsqueeze(0), fs_in=16000, fs_out=48000)["waveform"]
    a = out_a.squeeze(0).cpu().numpy()

    # (B) 16kHz 콘텐츠를 48kHz로 미리 업샘플해서 입력
    x48 = AF.resample(x16, 16000, 48000)
    report("input(pre-upsampled 48k)", x48.numpy(), 48000)
    out_b = model.process_waveform(x48.unsqueeze(0), fs_in=48000, fs_out=48000)["waveform"]
    b = out_b.squeeze(0).cpu().numpy()

    report("A: fs_in=16000 (native)", a, 48000)
    report("B: fs_in=48000 (pre-upsampled)", b, 48000)

    tag = "on" if enable_band_detect else "off"
    sf.write(f"/tmp/claude-1003/-home-nas-user-Uihyeop-NN-Zoo-TF-Restormer-release/b1b8b161-6304-428e-b225-c6ba6cd3c4d3/scratchpad/out_A_native_{tag}.wav", a, 48000)
    sf.write(f"/tmp/claude-1003/-home-nas-user-Uihyeop-NN-Zoo-TF-Restormer-release/b1b8b161-6304-428e-b225-c6ba6cd3c4d3/scratchpad/out_B_upsampled_{tag}.wav", b, 48000)


if __name__ == "__main__":
    main(enable_band_detect=(len(sys.argv) > 1 and sys.argv[1] == "on"))
