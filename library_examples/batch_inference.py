"""batch_inference.py — Enhance all .wav files in a directory.

Iterates over every .wav file in an input directory, applies speech
enhancement, and writes results to an output directory (preserving
filenames).

Usage::

    python library_examples/batch_inference.py \\
        --input_dir  /data/noisy/ \\
        --output_dir /data/enhanced/ \\
        --config baseline.yaml \\
        --checkpoint path/to/checkpoints/baseline/

Files that fail to process are reported at the end; all other files are
processed regardless.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF_Restormer — batch-enhance all .wav files in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        metavar="DIR",
        help="Directory containing noisy .wav files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="Directory where enhanced .wav files will be written.",
    )
    parser.add_argument(
        "--config",
        default="baseline.yaml",
        metavar="CONFIG",
        help="Config name or absolute path to YAML (default: baseline.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Checkpoint directory (containing model.pt) or direct .pt file path. "
            "Omit to use the default exported location inside the package."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        metavar="DEVICE",
        help="PyTorch device string (default: cuda).",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        metavar="GLOB",
        help="Glob pattern to match input files (default: '*.wav').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Collect input files
    # ------------------------------------------------------------------
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(input_dir.glob(args.pattern))
    if not wav_files:
        print(f"[WARNING] No files matching '{args.pattern}' found in {input_dir}.")
        sys.exit(0)

    print(f"Found {len(wav_files)} file(s) in {input_dir}")

    # ------------------------------------------------------------------
    # 2. Prepare output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Load model (once, reused for all files)
    # ------------------------------------------------------------------
    from tf_restormer import SEInference  # noqa: PLC0415

    print(f"Loading model  : config={args.config!r}, device={args.device!r}")
    model = SEInference.from_pretrained(
        config=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # 4. Process each file
    # ------------------------------------------------------------------
    failed: list[tuple[Path, str]] = []
    t_start = time.perf_counter()

    for idx, wav_path in enumerate(wav_files, start=1):
        out_path = output_dir / wav_path.name
        print(f"[{idx:3d}/{len(wav_files)}] {wav_path.name}", end=" ... ", flush=True)
        try:
            model.process_file(
                input_path=str(wav_path),
                output_path=str(out_path),
            )
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")
            failed.append((wav_path, str(exc)))

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    n_ok = len(wav_files) - len(failed)
    print(f"\nCompleted {n_ok}/{len(wav_files)} file(s) in {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")

    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for path, msg in failed:
            print(f"  {path.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
