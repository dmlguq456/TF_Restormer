"""
main_infer.py
~~~~~~~~~~~~~
CLI entry point for TF_Restormer inference-only workflows.

Responsibilities
----------------
* Load config / checkpoint.
* Build EngineInfer (tensor-in / tensor-out) or EngineEval.
* Dispatch to one of three inference paths:
    1. eval mode          → EngineEval.run_eval() per testset_key
    2. folder / file mode → _infer_directory() or _infer_file()
    3. dataloader mode    → _infer_dataloader() per testset_key

All file I/O lives in this module; EngineInfer itself is I/O-free.
"""

from __future__ import annotations

import argparse
import os

import soundfile as sf
import torch
from glob import glob
from loguru import logger
from tqdm import tqdm
from torchaudio.functional import resample as torch_resample

from .model import Model
from .dataset import get_dataloaders
from .engine_infer import EngineInfer
from tf_restormer.utils import util_engine
from tf_restormer.utils.decorators import logger_wraps


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_inference(
    args: argparse.Namespace,
    config: dict,
    config_name: str,
    fs_src: int = 16000,
    fs_in: int = 16000,
) -> EngineInfer:
    """Create model, load checkpoint, and build an EngineInfer.

    Args:
        args:        CLI args (used for gpuid and checkpoint path discovery).
        config:      Full experiment config dict.
        config_name: YAML file name (e.g. "baseline.yaml") — used to
                     reconstruct the checkpoint directory path.
        fs_src:      Output sample rate expected from the model (Hz).
        fs_in:       Input sample rate the model expects (Hz).

    Returns:
        A fully initialised EngineInfer with the latest checkpoint loaded.
    """
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(",")))
    device = torch.device(f"cuda:{gpuid[0]}")

    model = Model(**config["model"])
    model = model.to(device)

    train_phase = config["train_phase"] + "_" + config["dataset_phase"]
    log_base = f"log/log_{train_phase}_{config_name}"
    chkp_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), log_base, "weights"
    )
    os.makedirs(chkp_path, exist_ok=True)

    # model-only load — no optimizer state needed for inference
    util_engine.load_last_checkpoint_n_get_epoch_model_only(
        chkp_path, model, location=device
    )

    engine = EngineInfer(
        config=config,
        model=model,
        device=device,
        fs_src=fs_src,
        fs_in=fs_in,
    )
    return engine


# ---------------------------------------------------------------------------
# Per-file inference
# ---------------------------------------------------------------------------

def _infer_file(
    engine: EngineInfer,
    file_path: str,
    output_dir: str,
    fs_in: int,
    fs_out: int,
) -> None:
    """Enhance a single audio file and write the result to output_dir.

    The output file preserves the original file name.

    Args:
        engine:     Initialised EngineInfer.
        file_path:  Absolute path to the input .wav/.flac file.
        output_dir: Directory where the enhanced file will be written.
        fs_in:      Sample rate the model expects as input (Hz).
        fs_out:     Sample rate for the output file (Hz).
    """
    os.makedirs(output_dir, exist_ok=True)

    wav_np, orig_fs = sf.read(file_path, dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np[:, 0]  # keep first channel
    wav = torch.from_numpy(wav_np)

    # Resample to model input rate if needed
    if orig_fs != fs_in:
        wav = torch_resample(wav, orig_fs, fs_in)

    result = engine.infer_session(wav, fs_in=fs_in, fs_out=fs_out)
    enhanced = result["waveform"].squeeze(0).cpu()  # (L,)

    out_name = os.path.splitext(os.path.basename(file_path))[0] + ".wav"
    out_path = os.path.join(output_dir, out_name)
    sf.write(out_path, enhanced.numpy(), fs_out)
    logger.info(f"[FileInfer] saved: {out_path}")


# ---------------------------------------------------------------------------
# Directory (folder) inference
# ---------------------------------------------------------------------------

def _infer_directory(
    engine: EngineInfer,
    input_dir: str,
    output_dir: str,
    fs_in: int,
    fs_out: int,
) -> None:
    """Enhance all .wav/.flac files under input_dir and write to output_dir.

    Sub-directory structure is preserved.

    Args:
        engine:     Initialised EngineInfer.
        input_dir:  Root directory to search for audio files.
        output_dir: Root directory for enhanced output files.
        fs_in:      Sample rate the model expects as input (Hz).
        fs_out:     Sample rate for the output files (Hz).
    """
    wav_files = sorted(glob(os.path.join(input_dir, "**", "*.wav"), recursive=True))
    wav_files += sorted(glob(os.path.join(input_dir, "**", "*.flac"), recursive=True))

    if not wav_files:
        logger.warning(f"No .wav/.flac files found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm(
        total=len(wav_files),
        unit="file",
        bar_format="{l_bar}{bar:5}{r_bar}{bar:-10b}",
        colour="CYAN",
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for in_path in wav_files:
            # Preserve relative directory structure
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            # Force .wav extension regardless of original format
            out_path = os.path.splitext(out_path)[0] + ".wav"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                wav_np, orig_fs = sf.read(in_path, dtype="float32")
                if wav_np.ndim > 1:
                    wav_np = wav_np[:, 0]
                wav = torch.from_numpy(wav_np)

                # Resample to model input rate if needed
                if orig_fs != fs_in:
                    wav = torch_resample(wav, orig_fs, fs_in)

                result = engine.infer_session(wav, fs_in=fs_in, fs_out=fs_out)
                enhanced = result["waveform"].squeeze(0).cpu()  # (L,)

                # Normalise peak to avoid clipping
                peak = torch.max(torch.abs(enhanced))
                if peak > 1e-8:
                    enhanced = enhanced / peak

                sf.write(out_path, enhanced.numpy(), fs_out)
                logger.debug(f"  saved: {out_path}")

            except Exception as exc:
                logger.error(f"Failed on {in_path}: {exc}")

            pbar.update(1)

    pbar.close()
    logger.info(f"[FolderInfer] Done. {len(wav_files)} files → {output_dir}")


# ---------------------------------------------------------------------------
# DataLoader-based inference
# ---------------------------------------------------------------------------

def _infer_dataloader(
    engine: EngineInfer,
    dataloader,
    enhanced_dump_path: str,
    input_dump_path: str,
) -> None:
    """Save enhanced and noisy-reference WAV files for every batch in a DataLoader.

    Per-batch fields used:
        * ``noisy_distort_input`` (float32) — fed to the model
        * ``noisy_distort``       (float32) — noisy reference saved as-is
        * ``file_name``           (list[str])
        * ``fs_in``               (int, scalar) — model input rate
        * ``fs_src``              (int, scalar) — model output / source rate

    Args:
        engine:             Initialised EngineInfer.
        dataloader:         PyTorch DataLoader yielding the above keys.
        enhanced_dump_path: Directory for enhanced output WAV files.
        input_dump_path:    Directory for noisy reference WAV files.
    """
    os.makedirs(enhanced_dump_path, exist_ok=True)
    os.makedirs(input_dump_path, exist_ok=True)

    engine.model.eval()
    pbar = tqdm(
        total=len(dataloader),
        unit="utt",
        bar_format="{l_bar}{bar:5}{r_bar}{bar:-10b}",
        colour="WHITE",
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for batch in dataloader:
            noisy_orig = batch["noisy_distort_input"].type(torch.float32)  # (B, L)
            noisy = batch["noisy_distort"].type(torch.float32)              # (B, L)
            file_name = batch["file_name"]
            fs_in = batch["fs_in"].item()
            fs_src = batch["fs_src"].item()

            # infer_chunk processes one utterance at a time; batch size assumed 1
            # (consistent with original sf.write(... file_name[0] ...) L1027-1028)
            result = engine.infer_chunk(noisy_orig, fs_in=fs_in, fs_out=fs_src)
            out_wav = result["waveform"]  # (1, L)

            # Write enhanced and noisy reference WAV files
            sf.write(
                os.path.join(enhanced_dump_path, f"{file_name[0]}.wav"),
                out_wav[0].cpu().numpy(),
                fs_src,
            )
            sf.write(
                os.path.join(input_dump_path, f"{file_name[0]}.wav"),
                noisy[0].cpu().numpy(),
                fs_src,
            )

            pbar.update(1)

    pbar.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@logger_wraps()
def main_infer(args: argparse.Namespace) -> None:
    """CLI entry point for inference-only workflows.

    Dispatch logic:
        1. args.engine_mode == "eval"  → EngineEval.run_eval() per testset_key
        2. args.input is set           → folder or single-file inference
        3. default                     → dataloader inference per testset_key

    Config loading follows the same pattern as main.py.
    """
    # ---- Config loading (mirrors main.py) ----
    from tf_restormer._config import load_config
    config_name = getattr(args, "config", "baseline.yaml")
    yaml_dict = load_config("TF_Restormer", config_name)
    logger.info(f"Using config file: {config_name}")
    config = yaml_dict["config"]

    # ---- GPU / device ----
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(",")))
    device = torch.device(f"cuda:{gpuid[0]}")

    # ==================================================================
    # Path 1 — evaluation mode (metrics + logging, no WAV dump required)
    # ==================================================================
    if args.engine_mode == "eval":
        from .engine_eval import EngineEval  # lazy: only imported when eval is requested
        testset_keys = config["dataset_test"]["testset_key"]
        if isinstance(testset_keys, str):
            testset_keys = [testset_keys]

        model_e = Model(**config["model"])

        for i, key in enumerate(testset_keys):
            logger.info(f"===== [{i+1}/{len(testset_keys)}] testset_key: \"{key}\" =====")
            config["dataset_test"]["testset_key"] = key
            dataloaders = get_dataloaders(
                args,
                config["dataset_phase"],
                config["dataset_test"],
                config["dataloader"],
            )
            engine = EngineEval(args, config, model_e, dataloaders, gpuid, device)
            engine.run_eval()
        # Clear metric model cache after ALL testsets complete — NOT inside run_eval()
        # to avoid reloading GPU models (WVMOS, UTMOS, etc.) between testsets.
        from tf_restormer.utils.metrics import _model_cache
        _model_cache.clear()
        return

    # ==================================================================
    # Path 2 — folder / single-file inference (no DataLoader)
    # ==================================================================
    input_path = getattr(args, "input", None)
    if input_path:
        # Resolve sample rates from config (use first testset_key)
        testset_keys = config["dataset_test"]["testset_key"]
        if isinstance(testset_keys, list):
            first_key = testset_keys[0]
        else:
            first_key = testset_keys
        fs_src = config["dataset_test"][first_key]["sample_rate_src"]
        fs_in = config["dataset_test"][first_key]["sample_rate_in"]

        output_path = getattr(args, "output", None) or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "inference_wav", "folder_output"
        )

        logger.info(f"[FolderInfer] input={input_path}  output={output_path}")

        engine = setup_inference(args, config, config_name, fs_src=fs_src, fs_in=fs_in)

        if os.path.isdir(input_path):
            _infer_directory(engine, input_path, output_path, fs_in=fs_in, fs_out=fs_src)
        else:
            _infer_file(engine, input_path, output_path, fs_in=fs_in, fs_out=fs_src)
        return

    # ==================================================================
    # Path 3 — DataLoader inference per testset_key
    # Model is created once and reused; EngineInfer is recreated per key
    # (cheap: no weights re-loaded, only STFT keys differ if fs changes)
    # ==================================================================
    testset_keys = config["dataset_test"]["testset_key"]
    if isinstance(testset_keys, str):
        testset_keys = [testset_keys]

    # Build model once outside the loop
    model_e = Model(**config["model"])

    # Load checkpoint once — reuse the same model across testset_keys
    train_phase = config["train_phase"] + "_" + config["dataset_phase"]
    log_base = f"log/log_{train_phase}_{config_name}"
    chkp_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), log_base, "weights"
    )
    os.makedirs(chkp_path, exist_ok=True)
    util_engine.load_last_checkpoint_n_get_epoch_model_only(
        chkp_path, model_e, location=device
    )
    model_e = model_e.to(device)

    for i, key in enumerate(testset_keys):
        logger.info(f"===== [{i+1}/{len(testset_keys)}] testset_key: \"{key}\" =====")
        config["dataset_test"]["testset_key"] = key

        fs_src = config["dataset_test"][key]["sample_rate_src"]
        fs_in = config["dataset_test"][key]["sample_rate_in"]

        dataloaders = get_dataloaders(
            args,
            config["dataset_phase"],
            config["dataset_test"],
            config["dataloader"],
        )

        # EngineInfer is lightweight to construct (no checkpoint I/O here)
        engine = EngineInfer(
            config=config,
            model=model_e,
            device=device,
            fs_src=fs_src,
            fs_in=fs_in,
        )

        # Resolve dump paths
        if getattr(args, "dump_path", None) is not None:
            enhanced_dump_path = os.path.join(
                args.dump_path,
                "inference_wav",
                f"{key}_{train_phase}_{config_name[:-5]}_{str(fs_in)[:-3]}kto{str(fs_src)[:-3]}k",
            )
            input_dump_path = os.path.join(
                args.dump_path,
                "inference_wav",
                f"{key}_input_{str(fs_in)[:-3]}kto{str(fs_src)[:-3]}k",
            )
        else:
            enhanced_dump_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "inference_wav",
                f"{key}_{train_phase}_{config_name[:-5]}_{str(fs_in)[:-3]}kto{str(fs_src)[:-3]}k",
            )
            input_dump_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "inference_wav",
                f"{key}_input_{str(fs_in)[:-3]}kto{str(fs_src)[:-3]}k",
            )

        logger.info(f"Inference dump path: {enhanced_dump_path}")

        _infer_dataloader(
            engine,
            dataloaders["test"],
            enhanced_dump_path,
            input_dump_path,
        )

        logger.info(
            f"Inference and file writing for {key} dataset done!"
        )
