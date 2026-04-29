"""streaming_inference.py — Chunk-by-chunk streaming speech enhancement.

Streaming inference requires the online (Mamba) model.
Install: uv sync --extra mamba

This example simulates a real-time stream by reading an audio file in
small chunks (4096 samples by default) and feeding them to an
InferenceSession one at a time — exactly as a microphone callback would.

WHY streaming.yaml (not baseline.yaml)?
----------------------------------------
Streaming mode requires a *causal* model — one where each output frame
depends only on past and present inputs, never future ones.

  - baseline.yaml  : online=False  (Transformer, bi-directional attention)
  - streaming.yaml : online=True   (Mamba-based, causal / online-ready)

Using baseline.yaml in streaming mode runs a non-causal model on causal
chunks, producing meaningless results because the model expects to see the
full context.  Always use a config with model.online=True for streaming.

Usage::

    python library_examples/streaming_inference.py \\
        --input  noisy.wav \\
        --output enhanced_stream.wav \\
        --config streaming.yaml \\
        --checkpoint path/to/checkpoints/streaming/

Key streaming API pattern::

    session = model.create_session(fs_in=input_sample_rate, streaming=True)
    for chunk in audio_chunks:
        results = session.feed_waveform(chunk)   # consume immediately
        for r in results:
            play_audio(r["waveform"])            # or accumulate
    drained, tail = session.flush()              # drain remaining buffer
    if tail is not None:
        play_audio(tail["waveform"])

IMPORTANT — consume feed_waveform() results immediately:
    In streaming mode, each feed_waveform() call returns one result dict
    per completed internal chunk.  Do NOT discard these results and collect
    only flush() output (unlike batch mode which uses finalize()).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF_Restormer — streaming (chunk-by-chunk) speech enhancement",
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
        default="enhanced_stream.wav",
        metavar="FILE",
        help="Path for the enhanced output .wav file (default: enhanced_stream.wav).",
    )
    parser.add_argument(
        "--config",
        default="streaming.yaml",
        metavar="CONFIG",
        help=(
            "Config name or absolute path to YAML. "
            "MUST use a config with model.online=True (e.g. streaming.yaml). "
            "Default: streaming.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Checkpoint directory or .pt file path.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        metavar="DEVICE",
        help="PyTorch device string (default: cuda).",
    )
    parser.add_argument(
        "--chunk_samples",
        type=int,
        default=4096,
        metavar="N",
        help=(
            "Number of audio samples per feed_waveform() call. "
            "Simulates real-time microphone buffer size (default: 4096)."
        ),
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
    # 2. Load audio
    # ------------------------------------------------------------------
    try:
        import soundfile as sf
    except ImportError:
        print(
            "[ERROR] soundfile is required. Install: pip install soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch

    wav_np, file_sr = sf.read(str(input_path), dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np[:, 0]  # stereo -> mono (first channel)
    waveform = torch.from_numpy(wav_np)
    print(f"Loaded  : {input_path.name}  ({len(wav_np)} samples @ {file_sr} Hz)")

    # ------------------------------------------------------------------
    # 3. Load model
    #    MUST use a config with model.online=True (e.g. streaming.yaml).
    #    baseline.yaml (online=False) is NOT suitable for streaming.
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
    # 4. Create a streaming session and process chunks
    #
    # streaming=True: model processes each chunk independently (causal).
    # Each feed_waveform() call returns one dict per completed internal
    # chunk — consume these results immediately, do NOT discard them.
    # ------------------------------------------------------------------
    session = model.create_session(fs_in=file_sr, streaming=True)

    chunk_size = args.chunk_samples
    n_total = waveform.shape[0]
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    enhanced_parts: list[torch.Tensor] = []

    print(f"Streaming      : {n_chunks} chunk(s) of {chunk_size} samples each")
    for i in range(0, n_total, chunk_size):
        chunk = waveform[i : i + chunk_size]  # shape: (N,) where N <= chunk_size

        # feed_waveform() returns a list — one dict per completed internal chunk.
        # In streaming mode, consume these results immediately (not via finalize()).
        results = session.feed_waveform(chunk)
        for r in results:
            enhanced_parts.append(r["waveform"].squeeze(0).cpu())

    # ------------------------------------------------------------------
    # 5. Flush remaining buffered samples
    #
    # flush() returns (drained_list, tail_dict).
    #   drained: results from any complete chunks still in the buffer.
    #   tail:    the final partial chunk, or None if the buffer was empty.
    # ------------------------------------------------------------------
    drained, tail = session.flush()
    for r in drained:
        enhanced_parts.append(r["waveform"].squeeze(0).cpu())
    if tail is not None:
        enhanced_parts.append(tail["waveform"].squeeze(0).cpu())

    # ------------------------------------------------------------------
    # 6. Concatenate and save
    # ------------------------------------------------------------------
    if not enhanced_parts:
        print("[WARNING] No enhanced output produced.", file=sys.stderr)
        sys.exit(1)

    enhanced = torch.cat(enhanced_parts, dim=0).numpy()
    out_sr = model._fs_src  # native output sample rate from config (e.g. 48000)

    sf.write(args.output, enhanced, out_sr)
    print(f"\nOutput written : {args.output}")
    print(f"Output rate    : {out_sr} Hz")
    print(f"Duration       : {len(enhanced) / out_sr:.2f} s")
    print(f"Enhanced parts : {len(enhanced_parts)} chunk(s) accumulated")


if __name__ == "__main__":
    main()
