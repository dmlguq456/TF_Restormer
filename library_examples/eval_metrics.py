"""eval_metrics.py — Evaluate speech enhancement quality with built-in metrics.

Demonstrates how to use the tf_restormer.utils.metrics module to compute
quality scores independently of the training pipeline.

Available metrics:
  Intrusive (need clean reference):
    pesq, stoi, sdr, lsd, mcd, composite
  Non-intrusive (no reference needed):
    dnsmos, dnsmos_sig, dnsmos_bak, nisqa
  Neural perceptual:
    utmos, wvmos
  Semantic / ASR:
    bleu, bertscore, tokendist, wer_whisper, wer_w2v, cer_whisper

Installation of extras:
  uv sync --extra metrics-intrusive    # pesq, stoi, sdr, lsd, mcd, composite
  uv sync --extra metrics-nonintrusive # dnsmos, nisqa
  uv sync --extra metrics-neural       # utmos, wvmos, wer_*, cer_*
  uv sync --extra metrics-semantic     # bleu, bertscore, tokendist

Usage::

    # Compare enhanced vs clean (intrusive metrics)
    python library_examples/eval_metrics.py \\
        --enhanced enhanced.wav \\
        --clean    clean.wav

    # Score enhanced file alone (non-intrusive)
    python library_examples/eval_metrics.py \\
        --enhanced enhanced.wav

    # Choose specific metrics
    python library_examples/eval_metrics.py \\
        --enhanced enhanced.wav --clean clean.wav \\
        --metrics pesq stoi dnsmos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF_Restormer — compute speech quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--enhanced",
        required=True,
        metavar="FILE",
        help="Path to the enhanced (estimated) audio file.",
    )
    parser.add_argument(
        "--clean",
        default=None,
        metavar="FILE",
        help=(
            "Path to the clean (reference) audio file. "
            "Required for intrusive metrics (pesq, stoi, sdr, ...)."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="KEY",
        help=(
            "One or more metric keys to compute. "
            "Defaults: pesq + stoi (intrusive, if --clean given) + dnsmos. "
            "Use 'list' to print all available keys and exit."
        ),
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        metavar="HZ",
        help=(
            "Sample rate to use when calling metrics. "
            "Defaults to the sample rate of the enhanced file."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="Device for neural metrics (utmos, wvmos, dnsmos neural). Default: cpu.",
    )
    return parser.parse_args()


def load_audio(path: Path) -> tuple["torch.Tensor", int]:
    """Read a WAV file into a 1-D float32 tensor.

    Returns:
        (waveform, sample_rate)  — waveform shape: (L,)
    """
    try:
        import soundfile as sf
    except ImportError:
        print(
            "[ERROR] soundfile is required. Install: pip install soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch

    wav_np, sr = sf.read(str(path), dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np[:, 0]  # stereo -> mono
    return torch.from_numpy(wav_np), sr


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Handle "list" request
    # ------------------------------------------------------------------
    from tf_restormer.utils.metrics import compute_metric, list_metrics  # noqa: PLC0415

    if args.metrics and args.metrics[0].lower() == "list":
        print("Available metric keys:")
        for key in list_metrics():
            print(f"  {key}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # 1. Validate and load audio
    # ------------------------------------------------------------------
    enhanced_path = Path(args.enhanced)
    if not enhanced_path.is_file():
        print(f"[ERROR] Enhanced file not found: {enhanced_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Enhanced file  : {enhanced_path}")
    estim_wav, estim_sr = load_audio(enhanced_path)

    target_wav = None
    target_sr = None
    if args.clean is not None:
        clean_path = Path(args.clean)
        if not clean_path.is_file():
            print(f"[ERROR] Clean file not found: {clean_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Clean file     : {clean_path}")
        target_wav, target_sr = load_audio(clean_path)

    # Resolve evaluation sample rate
    fs = args.fs if args.fs is not None else estim_sr
    print(f"Eval sample rate: {fs} Hz\n")

    # ------------------------------------------------------------------
    # 2. Determine which metrics to compute
    # ------------------------------------------------------------------
    if args.metrics is not None:
        requested = args.metrics
    else:
        # Default set: intrusive metrics require clean reference
        requested = []
        if target_wav is not None:
            requested += ["pesq", "stoi"]
        requested += ["dnsmos"]

    # ------------------------------------------------------------------
    # 3. Compute each metric
    # ------------------------------------------------------------------
    results: dict[str, object] = {}

    for key in requested:
        # Decide whether to pass target (intrusive metrics need it)
        need_target = key in {
            "pesq", "stoi", "sdr", "lsd", "mcd", "composite",
        }

        if need_target and target_wav is None:
            print(f"  [{key}] SKIPPED — intrusive metric requires --clean file")
            continue

        try:
            if need_target:
                score = compute_metric(key, estim_wav, target=target_wav, fs=fs)
            else:
                score = compute_metric(key, estim_wav, fs=fs)

            results[key] = score
            # Format output depending on type
            if isinstance(score, (int, float)):
                print(f"  {key:<20s} : {score:.4f}")
            elif isinstance(score, dict):
                # e.g. composite returns {"csig": ..., "cbak": ..., "covl": ...}
                print(f"  {key:<20s} : {score}")
            else:
                print(f"  {key:<20s} : {score}")

        except ImportError as exc:
            # Extra not installed — show a helpful install hint
            from tf_restormer.utils.metrics import get_extra_name  # noqa: PLC0415

            extra = get_extra_name(key)
            print(
                f"  [{key}] SKIPPED — optional dependency missing.\n"
                f"           Install: uv sync --extra {extra}\n"
                f"           (or: pip install tf-restormer[{extra}])\n"
                f"           Details: {exc}"
            )
        except KeyError as exc:
            print(
                f"  [{key}] ERROR — unknown metric key: {exc}\n"
                f"           Run with --metrics list to see all available keys."
            )
        except Exception as exc:
            print(f"  [{key}] ERROR — {exc}")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    n_ok = len(results)
    n_req = len(requested)
    print(f"\nComputed {n_ok}/{n_req} metric(s) successfully.")

    if not results:
        print(
            "No metrics could be computed.  "
            "Check that the required extras are installed (see above)."
        )


if __name__ == "__main__":
    main()
