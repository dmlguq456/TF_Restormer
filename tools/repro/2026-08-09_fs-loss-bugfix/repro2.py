"""Issue #2 정밀 재현: native 16k vs pre-upsampled 48k, band_detect on/off."""
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tf_restormer import SEInference

OUT = "/tmp/claude-1003/-home-nas-user-Uihyeop-NN-Zoo-TF-Restormer-release/b1b8b161-6304-428e-b225-c6ba6cd3c4d3/scratchpad"
CKPT = "tf_restormer/checkpoints/baseline"
SAMPLES = [
    "data/valid_sample/UNIVERSE_sample/0015.wav",
    "data/valid_sample/Real_sample/0.wav",
]


def avg_spectrum(wav, sr, nfft=2048):
    x = wav.astype(np.float64)
    hop = nfft // 2
    frames = [x[i:i + nfft] for i in range(0, max(1, len(x) - nfft), hop)]
    w = np.hanning(nfft)
    P = np.mean([np.abs(np.fft.rfft(f * w)) ** 2 for f in frames], axis=0)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    return freqs, P


def band_report(tag, wav, sr):
    f, P = avg_spectrum(wav, sr)
    def E(lo, hi):
        m = (f >= lo) & (f < hi)
        return P[m].mean() if m.any() else 0.0
    lf = E(300, 3400)
    b1 = E(8200, 12000)
    b2 = E(12000, 20000)
    print(f"  [{tag:34s}] 8.2-12k: {10*np.log10((b1+1e-30)/(lf+1e-30)):+6.1f} dB   "
          f"12-20k: {10*np.log10((b2+1e-30)/(lf+1e-30)):+6.1f} dB  (rel. to 0.3-3.4k)")
    return f, P


def run(model, x, fs_in, bd_enable):
    model.engine.model.band_detect_enable = bd_enable
    out = model.process_waveform(x.unsqueeze(0), fs_in=fs_in, fs_out=48000)
    return out["waveform"].squeeze(0).cpu().numpy(), out


def main():
    model = SEInference.from_pretrained(checkpoint_path=CKPT, device="cuda:0")
    inner = model.engine.model
    print(f"loaded: band_detect_enable(config default) = {inner.band_detect_enable}, "
          f"db_below_peak = {inner.band_db_below_peak}, up.min_F = {inner.up.min_F}, up.max_F = {inner.up.max_F}")
    print(f"model fs_out = {model.fs_out}\n")

    for path in SAMPLES:
        clean, sr0 = sf.read(path, dtype="float32")
        if clean.ndim > 1:
            clean = clean[:, 0]
        t = torch.from_numpy(clean)
        x16 = AF.resample(t, sr0, 16000) if sr0 != 16000 else t
        x48 = AF.resample(x16, 16000, 48000)
        print(f"{path}  (src sr={sr0}, {len(x16)/16000:.2f}s)")
        band_in = band_report("input: 16k content @48k", x48.numpy(), 48000)

        a, oa = run(model, x16, 16000, False)
        b, _ = run(model, x48, 48000, False)
        c, _ = run(model, x48, 48000, True)
        print(f"    out lens: A={len(a)} B={len(b)} C={len(c)} (expect {len(x16)*3})")
        fa = band_report("A native 16k in  (bd off)", a, 48000)
        fb = band_report("B pre-upsampled  (bd OFF = HF)", b, 48000)
        fc = band_report("C pre-upsampled  (bd ON)", c, 48000)

        # 2kHz 단위 스펙트럼 프로파일 (0.3-3.4k 기준 상대 dB)
        print(f"    {'band(kHz)':>10s} " + " ".join(f"{lo:2d}-{lo+2:2d}" for lo in range(0, 24, 2)))
        for name, (f, P) in [("input", band_in), ("A native", fa), ("B bd-off", fb), ("C bd-on", fc)]:
            ref = P[(f >= 300) & (f < 3400)].mean()
            cells = []
            for lo in range(0, 24000, 2000):
                m = (f >= lo) & (f < lo + 2000)
                cells.append(f"{10*np.log10((P[m].mean()+1e-30)/(ref+1e-30)):+5.0f}")
            print(f"    {name:>10s} " + " ".join(cells))
        print()


if __name__ == "__main__":
    main()
