"""epoch.0020이 어느 체크포인트에서 이어졌는지 가중치 거리로 추적."""
import torch

L = "tf_restormer/models/TF_Restormer/log"
A = f"{L}/log_adversarial_to48k_baseline.yaml/weights"
P = f"{L}/log_pretrain_to48k_baseline.yaml/weights"


def sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.float() for k, v in ck["model_state_dict"].items() if v.dtype.is_floating_point}


def dist(a, b):
    num = sum(((a[k] - b[k]) ** 2).sum().item() for k in a if k in b)
    den = sum((a[k] ** 2).sum().item() for k in a)
    return (num / den) ** 0.5


ref = sd(f"{A}/epoch.0020.pth")
cands = [
    ("adv epoch.0019 (9월)", f"{A}/epoch.0019.pth"),
    ("adv epoch.0016 (9월)", f"{A}/epoch.0016.pth"),
    ("pretrain epoch.0015 (11월)", f"{P}/epoch.0015.pth"),
    ("pretrain epoch.0010 (12/19)", f"{P}/epoch.0010.pth"),
    ("adv epoch.0021 (12월, 참고)", f"{A}/epoch.0021.pth"),
]
print("epoch.0020 기준 상대 L2 거리 (작을수록 직전 상태):")
for name, p in cands:
    print(f"  {name:32s} {dist(ref, sd(p)):.6f}")

print("\n연속 epoch 간 상대 L2 거리:")
prev = None
for e in [16, 19, 20, 21, 22, 23, 24, 25]:
    cur = sd(f"{A}/epoch.{e:04d}.pth")
    if prev is not None:
        print(f"  epoch.{prev_e:04d} -> epoch.{e:04d}: {dist(cur, prev):.6f}")
    prev, prev_e = cur, e
