"""epoch별 loss 궤적 + lr + config 실질값 비교."""
import yaml
import torch

SC = "/tmp/claude-1003/-home-nas-user-Uihyeop-NN-Zoo-TF-Restormer-release/b1b8b161-6304-428e-b225-c6ba6cd3c4d3/scratchpad"
L = "tf_restormer/models/TF_Restormer/log/log_adversarial_to48k_baseline.yaml"

# ---- 1. config 실질값 비교 ----
a = yaml.safe_load(open(f"{SC}/cfg_sep24.yaml"))["config"]
b = yaml.safe_load(open("tf_restormer/checkpoints/baseline/config.yaml"))["config"]


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


fa, fb = flat(a.get("engine")), flat(b.get("engine"))
diffs = [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
print("=== engine 설정 실질 차이 (9/24 커밋 vs 4/15 export 스냅샷) ===")
if not diffs:
    print("  (없음 — 값 동일)")
for k, v1, v2 in diffs:
    if "sample_validation" in k:
        continue
    print(f"  {k}: {v1}  ->  {v2}")

fa2, fb2 = flat({k: v for k, v in a.items() if k in ("model", "stft", "dataset")}), \
           flat({k: v for k, v in b.items() if k in ("model", "stft", "dataset")})
d2 = [(k, fa2.get(k), fb2.get(k)) for k in sorted(set(fa2) | set(fb2)) if fa2.get(k) != fb2.get(k)]
print("=== model/stft/dataset 차이 ===")
print("  (없음 — 값 동일)" if not d2 else "")
for k, v1, v2 in d2:
    print(f"  {k}: {v1}  ->  {v2}")

# ---- 2. epoch별 loss / lr ----
print("\n=== adversarial 단계 epoch별 기록 ===")
print(f"{'epoch':>6s} {'train_loss':>12s} {'valid_loss':>12s} {'lr':>12s}  {'D lr':>12s}")
for e in [16, 19, 20, 21, 22, 23, 24, 25]:
    try:
        ck = torch.load(f"{L}/weights/epoch.{e:04d}.pth", map_location="cpu", weights_only=False)
    except FileNotFoundError:
        continue
    lr = ck["optimizer_state_dict"]["param_groups"][0].get("lr", float("nan"))
    try:
        ckd = torch.load(f"{L}/weights_D/epoch.{e:04d}.pth", map_location="cpu", weights_only=False)
        lrd = ckd["optimizer_state_dict"]["param_groups"][0].get("lr", float("nan"))
    except Exception:
        lrd = float("nan")
    print(f"{ck['epoch']:>6d} {ck['train_loss']:>12.5f} {ck['valid_loss']:>12.5f} {lr:>12.3e}  {lrd:>12.3e}")
