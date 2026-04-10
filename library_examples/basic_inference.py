"""basic_inference.py — Minimal single-file speech enhancement example.

Load a TF_Restormer model and enhance one audio file.

Usage::

    python library_examples/basic_inference.py \\
        --input  noisy.wav \\
        --output enhanced.wav \\
        --config baseline.yaml \\
        --checkpoint path/to/checkpoints/baseline/

The enhanced output is written at the model's native output sample rate
(48 kHz by default).  Pass --checkpoint as a directory that contains
model.pt, or a direct path to a .pt file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF_Restormer — enhance a single audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to the noisy input .wav file.",
    )
    parser.add_argument(
        "--output",
        default="enhanced.wav",
        metavar="FILE",
        help="Path for the enhanced output .wav file (default: enhanced.wav).",
    )
    parser.add_argument(
        "--config",
        default="baseline.yaml",
        metavar="CONFIG",
        help=(
            "Config name (e.g. 'baseline.yaml') or absolute path to a YAML file. "
            "Default: baseline.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Path to checkpoint directory (containing model.pt) or a .pt/.pth file. "
            "Omit to use the default exported location inside the package."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        metavar="DEVICE",
        help="PyTorch device string: 'cuda', 'cpu', 'cuda:1', etc. (default: cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    from tf_restormer import SEInference  # noqa: PLC0415

    print(f"Loading model  : config={args.config!r}, device={args.device!r}")
    model = SEInference.from_pretrained(
        config=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # 3. Enhance the file
    #
    # process_file() reads the audio, runs enhancement, and writes the
    # output if output_path is given.  The returned dict always contains:
    #   result["waveform"]    — (1, L_out) torch.Tensor at fs_out
    #   result["sample_rate"] — output sample rate (int)
    # ------------------------------------------------------------------
    print(f"Processing     : {input_path}")
    result = model.process_file(
        input_path=str(input_path),
        output_path=args.output,
    )

    out_sr = result.get("sample_rate", model._fs_src)
    out_len_sec = result["waveform"].shape[-1] / out_sr
    print(f"Output written : {args.output}")
    print(f"Output rate    : {out_sr} Hz")
    print(f"Duration       : {out_len_sec:.2f} s")


if __name__ == "__main__":
    main()
