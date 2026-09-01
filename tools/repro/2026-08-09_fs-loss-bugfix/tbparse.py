"""tensorboard 이벤트 파일을 의존성 없이 파싱해 scalar 궤적을 뽑는다."""
import struct
import sys
import glob
import os
from collections import defaultdict


def read_varint(buf, i):
    val, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def parse_fields(buf):
    """protobuf wire format -> {field_no: [raw values]}"""
    i, out = 0, defaultdict(list)
    n = len(buf)
    while i < n:
        try:
            key, i = read_varint(buf, i)
        except IndexError:
            break
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(buf, i)
            out[fno].append(v)
        elif wt == 1:
            out[fno].append(buf[i:i + 8]); i += 8
        elif wt == 2:
            ln, i = read_varint(buf, i)
            out[fno].append(buf[i:i + ln]); i += ln
        elif wt == 5:
            out[fno].append(buf[i:i + 4]); i += 4
        else:
            break
    return out


def iter_events(path):
    with open(path, "rb") as f:
        data = f.read()
    i, n = 0, len(data)
    while i + 12 <= n:
        (ln,) = struct.unpack_from("<Q", data, i)
        i += 12
        if i + ln + 4 > n:
            break
        rec = data[i:i + ln]
        i += ln + 4
        yield rec


def scalars(path):
    """-> [(step, tag, value)]"""
    out = []
    for rec in iter_events(path):
        ev = parse_fields(rec)
        step = ev.get(2, [0])[0]
        for s in ev.get(5, []):
            summ = parse_fields(s)
            for v in summ.get(1, []):
                val = parse_fields(v)
                tag = val.get(1, [b""])[0].decode("utf-8", "ignore")
                if 2 in val and len(val[2][0]) == 4:
                    out.append((step, tag, struct.unpack("<f", val[2][0])[0]))
    return out


if __name__ == "__main__":
    root = sys.argv[1]
    files = sorted(glob.glob(os.path.join(root, "**", "events.out.tfevents.*"), recursive=True))
    print(f"{len(files)} event files under {root}")
    per_tag = defaultdict(list)
    for fp in files:
        sub = os.path.relpath(os.path.dirname(fp), root)
        for step, tag, val in scalars(fp):
            key = tag if sub == "." else f"{sub}/{tag}"
            per_tag[key].append((step, val))
    for tag in sorted(per_tag):
        pts = sorted(per_tag[tag])
        if len(pts) < 2:
            continue
        steps = [p[0] for p in pts]
        print(f"\n### {tag}   ({len(pts)} points, step {min(steps)}~{max(steps)})")
        # step 구간별 평균
        lo, hi = min(steps), max(steps)
        nb = min(12, max(2, (hi - lo + 1)))
        width = max(1, (hi - lo + 1) // nb)
        buckets = defaultdict(list)
        for s, v in pts:
            buckets[(s - lo) // width].append(v)
        for b in sorted(buckets):
            vs = buckets[b]
            print(f"   step {lo + b*width:>7d}-{lo + (b+1)*width - 1:<7d} n={len(vs):<5d} mean={sum(vs)/len(vs):+.5f}")
