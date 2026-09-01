"""공개 체크포인트가 고주파 대역(mask token 경로)을 실제로 생성하는지 확인."""
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from tf_restormer import SEInference

CKPT = "tf_restormer/checkpoints/baseline"

model = SEInference.from_pretrained(checkpoint_path=CKPT, device="cuda:0")
m = model.engine.model
m.band_detect_enable = False

mt = m.up.mask_token
print(f"up.mask_token: shape={tuple(mt.shape)}  mean={mt.mean():.4e}  std={mt.std():.4e}  "
      f"absmax={mt.abs().max():.4e}  all_zero={bool((mt == 0).all())}")

clean, sr0 = sf.read("data/valid_sample/UNIVERSE_sample/0015.wav", dtype="float32")
if clean.ndim > 1:
    clean = clean[:, 0]
x16 = torch.from_numpy(clean) if sr0 == 16000 else AF.resample(torch.from_numpy(clean), sr0, 16000)

stft16 = model.engine.stft["16000"]
X = stft16(x16.unsqueeze(0).cuda(), cplx=True)      # (1, F=321, T)
print(f"input STFT: F={X.shape[1]}  T={X.shape[2]}")

xin = torch.stack([X.real, X.imag], dim=-1)

with torch.inference_mode():
    for out_F, label in [(321, "out_F=321 (16k out)"), (961, "out_F=961 (48k out)")]:
        y = m(xin, out_F=out_F)                     # (1, out_F, T, 2)
        mag = torch.sqrt(y[..., 0] ** 2 + y[..., 1] ** 2).mean(dim=(0, 2))  # (out_F,)
        lo = mag[:321].mean().item()
        print(f"\n{label}: y.shape={tuple(y.shape)}")
        print(f"   mean|Y| bins 0-320   = {lo:.4e}")
        if out_F > 321:
            hi = mag[321:].mean().item()
            print(f"   mean|Y| bins 321-960 = {hi:.4e}   ({20*np.log10((hi+1e-30)/(lo+1e-30)):+.1f} dB rel.)")
            # 대역별로 세분
            for s in range(321, 961, 160):
                e = min(s + 160, 961)
                seg = mag[s:e].mean().item()
                f0, f1 = s * 25, e * 25
                print(f"      bins {s:3d}-{e-1:3d} ({f0/1000:.1f}-{f1/1000:.1f} kHz): "
                      f"{20*np.log10((seg+1e-30)/(lo+1e-30)):+7.1f} dB")

# 디코더 직전 단계까지 추적: 마스크 토큰이 디코더를 통과한 뒤 살아남는지
with torch.inference_mode():
    x = xin
    xn, xs = m.norm(x)
    x_enc = m.input_embed(xn)
    B, F, T, C = x_enc.shape
    pos_f = m.sinusoids(F, C).reshape(1, F, 1, C).to(x_enc.device)
    x_enc = x_enc + pos_f
    x_enc = m.encoder(x_enc)
    x_dec, pad_len = m.up(x_enc, 961)
    print(f"\nafter FreqUpsampleToken: x_dec.shape={tuple(x_dec.shape)} pad_len={pad_len}")
    print(f"   |x_dec| bins 0-320   = {x_dec[:, :321].abs().mean():.4e}")
    print(f"   |x_dec| bins 321-960 = {x_dec[:, 321:].abs().mean():.4e}")
    x_dec2 = m.decoder(x_dec, x_enc, pad_len)
    print(f"after decoder stage:")
    print(f"   |h| bins 0-320   = {x_dec2[:, :321].abs().mean():.4e}")
    print(f"   |h| bins 321-960 = {x_dec2[:, 321:].abs().mean():.4e}")
    y = m.estimator(x_dec2)
    print(f"after estimator (pre-inorm):")
    print(f"   |Y| bins 0-320   = {y[:, :321].abs().mean():.4e}")
    print(f"   |Y| bins 321-960 = {y[:, 321:].abs().mean():.4e}")
