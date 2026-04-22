from __future__ import annotations

import json
import os
import re
import random
import shutil
import torch
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger


_EPOCH_PATTERN = re.compile(r'^epoch\.(\d+)\.(pt|pth|pkl)$')


class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, last_epoch: int = -1) -> None:
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            return 1.0
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)






def _fix_compiled_state_dict(state_dict: dict) -> dict:
    """Strip ``_orig_mod.`` prefix from keys produced by ``torch.compile()``.

    When a model is saved after wrapping with ``torch.compile()``, the state
    dict keys carry an ``_orig_mod.`` prefix (e.g. ``_orig_mod.encoder.weight``).
    This helper removes that prefix so the dict can be loaded into a plain
    (non-compiled) model.

    Args:
        state_dict: Raw state dict loaded from a checkpoint.

    Returns:
        A new dict with the prefix stripped from every key that has it.
        Keys without the prefix are kept unchanged.
    """
    return {
        k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
        for k, v in state_dict.items()
    }


def _find_latest_checkpoint(checkpoint_dir: str | "Path") -> tuple[str, int] | None:
    """Find the checkpoint file with the highest epoch number in *checkpoint_dir*.

    Primary search: files matching the training-checkpoint naming convention
    ``epoch.{NNNN}.pth`` (e.g. ``epoch.0016.pth``), ranked by epoch number.

    Fallback: if no ``epoch.*.pth`` files are found, returns the most recently
    modified ``.pt`` or ``.pth`` file in the directory (epoch reported as -1).

    Args:
        checkpoint_dir: Directory that may contain ``epoch.*.pth`` files.

    Returns:
        A ``(path, epoch)`` tuple for the best checkpoint found, or ``None``
        if the directory contains no ``.pt`` / ``.pth`` files at all.
    """
    from pathlib import Path as _Path
    directory = _Path(checkpoint_dir)
    pth_files = [
        f for f in os.listdir(directory)
        if f.endswith(".pth") and f.startswith("epoch.")
    ]
    if pth_files:
        epochs = [int(f.split('.')[1]) for f in pth_files]
        best_idx = epochs.index(max(epochs))
        path = str(directory / pth_files[best_idx])
        return path, epochs[best_idx]

    # Fallback: any .pt or .pth file, ranked by modification time
    candidates = list(directory.glob("*.pt")) + list(directory.glob("*.pth"))
    if candidates:
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(best), -1
    return None


def load_last_checkpoint_n_get_epoch(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, location: torch.device | str = 'cuda', fix_compiled: bool = False) -> int:
    """
    Load the latest checkpoint (model state and optimizer state) from a given directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model into which the checkpoint's model state should be loaded.
        optimizer (torch.optim.Optimizer | None): The optimizer into which the checkpoint's optimizer state should be loaded. Defaults to None (skip optimizer loading).
        location (str | torch.device, optional): Device location for loading the checkpoint. Defaults to 'cuda'.
        fix_compiled (bool): If True, strip ``_orig_mod.`` prefix from state dict
            keys before loading (needed when a checkpoint was saved from a
            ``torch.compile()``-wrapped model). Defaults to False.

    Returns:
        int: The epoch number associated with the loaded checkpoint.
             If no checkpoint is found, returns 1 as the starting epoch.

    Notes:
        - The checkpoint file is expected to have keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'.
        - If there are multiple checkpoint files in the directory, the one with the highest epoch number is loaded.
    """
    result = _find_latest_checkpoint(checkpoint_dir)

    # If no checkpoint files exist, return 1 as the starting epoch
    if result is None:
        return 1

    latest_checkpoint_file, found_epoch = result

    # Load the checkpoint into the model & optimizer
    logger.info(f"Loaded Pretrained model from {latest_checkpoint_file} .....")
    checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location, weights_only=True)
    model_state = checkpoint_dict['model_state_dict']
    if fix_compiled:
        model_state = _fix_compiled_state_dict(model_state)
    model.load_state_dict(model_state, strict=False)  # Depend on weight file's key!!
    if optimizer is not None and 'optimizer_state_dict' in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # Return next epoch; fall back to found_epoch when 'epoch' key is absent
    # (e.g. fallback .pth files that do not follow the training checkpoint format)
    saved_epoch = checkpoint_dict.get('epoch', found_epoch)
    return saved_epoch + 1


def save_checkpoint_per_nth(nth: int, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loss: float, valid_loss: float, checkpoint_path: str) -> None:
    """Save model and optimizer state every nth epoch.

    Args:
        nth: Interval for which checkpoints should be saved.
        epoch: The current training epoch.
        model: The model whose state needs to be saved.
        optimizer: The optimizer whose state needs to be saved.
        train_loss: Current training loss.
        valid_loss: Current validation loss.
        checkpoint_path: Directory path where the checkpoint will be saved.

    Returns:
        None
    """
    if epoch % nth == 0:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # Log and save the checkpoint file using wandb

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_parameters += param_count
        logger.info(f"{name}: {param_count}")
    logger.info(f"Total parameters: {(total_parameters / 1e6):.2f}M")

def model_params_mac_summary(model: torch.nn.Module, input_shape: tuple, metrics: list[str], device: torch.device) -> None:
    """Log model MACs and parameter counts using multiple profilers.

    Each profiler is independently optional: if the package is not installed
    an ImportError is caught and a warning is logged instead of crashing.
    This allows eval mode to run without train-only extras.
    """
    # ptflops
    if 'ptflops' in metrics:
        try:
            from ptflops import get_model_complexity_info
            MACs_ptflops, params_ptflops = get_model_complexity_info(model, (*input_shape,), print_per_layer_stat=False, verbose=False) # (num_samples,)
            MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
            logger.info(f"ptflops: MACs: {MACs_ptflops}, Params: {params_ptflops}")
        except ImportError:
            logger.info("Skipping ptflops profiling (not installed).")

    # thop
    if 'thop' in metrics:
        try:
            from thop import profile
            input = torch.randn(1, *input_shape).to(device)
            MACs_thop, params_thop = profile(model, inputs=input, verbose=False)
            MACs_thop, params_thop = MACs_thop/1e9, params_thop/1e6
            logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")
        except ImportError:
            logger.info("Skipping thop profiling (not installed).")

    # torchinfo
    if 'torchinfo' in metrics:
        try:
            from torchinfo import summary as summary_
            model_profile = summary_(model, input_size=(1, *input_shape), verbose=0, device=device)
            MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e9, model_profile.total_params/1e6
            logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")
        except ImportError:
            logger.info("Skipping torchinfo profiling (not installed).")


def create_sampler(dataset_size: int, num_samples: int) -> torch.utils.data.SubsetRandomSampler:
    indices = list(range(dataset_size))
    random.shuffle(indices)
    sampled_indices = indices[:num_samples]
    return torch.utils.data.SubsetRandomSampler(sampled_indices)


def create_dataloader_with_sampler(
    _dataloader: DataLoader,
    subset_conf: dict,
    loader_config: dict,
) -> DataLoader:
    """Return a new DataLoader with subset sampler if enabled, otherwise return the original.

    Args:
        _dataloader: The original DataLoader to wrap.
        subset_conf: A dict with keys "subset" (bool) and "num_per_epoch" (int).
        loader_config: DataLoader kwargs (batch_size, num_workers, pin_memory, drop_last).
    """
    if subset_conf["subset"]:
        sampler = create_sampler(len(_dataloader.dataset), subset_conf["num_per_epoch"])
        return DataLoader(
            dataset=_dataloader.dataset,
            collate_fn=_dataloader.collate_fn,
            sampler=sampler,
            **loader_config,
        )
    return _dataloader


def p_law_compress(x, c=0.3, mode=None):
    # mode : None / inverse
    x_abs = torch.clamp(x.abs(), min=1.0e-15)
    x_angle = x.angle()
    x_compressed = x_abs**(1/c) if mode == 'inverse' else x_abs**c
    return torch.polar(x_compressed, x_angle)


class RandSpecAugment:
    """SpecAugment for batched spectrograms with per-example independent masks.

    All range parameters are specified as [min, max] tuples:
      - t_len: time-mask length range
      - f_len: freq-mask length range
      - t_num: number of time masks per sample
      - f_num: number of freq masks per sample
    When per_example=True, each sample in the batch gets an independent mask.
    """
    def __init__(self,
                 t_len: tuple[int, int] = (10, 20), f_len: tuple[int, int] = (40, 60),
                 t_num: tuple[int, int] = (2,  5),  f_num: tuple[int, int] = (1, 2),
                 per_example: bool = True, device: str = "cpu") -> None:
        self.t_len, self.f_len = t_len, f_len
        self.t_num, self.f_num = t_num, f_num
        self.per_example = per_example
        self.device = device

    # -------- Sample non-overlapping 1-D intervals (pure python) --------
    @staticmethod
    def _rand_ranges(L: int, n_rng, w_rng):
        n = random.randint(*n_rng)
        w_min, w_max = w_rng
        ranges = []
        for _ in range(n):
            for _ in range(10):                     # retry up to 10 times
                w = random.randint(w_min, w_max)
                if w >= L:
                    continue
                s = random.randint(0, L - w)
                if all(not (s < e and s+w > b) for b, e in ranges):
                    ranges.append((s, s+w))
                    break
        return ranges

    # -------- Mask a single sample --------
    def _mask_single(self, spec: torch.Tensor, mask_idx:torch.Tensor) -> torch.Tensor:
        """
        spec : (F, T) tensor   –  returns masked copy
        """
        F, T = spec.shape[:2]
        out = spec.clone()

        # Time-mask
        for s, e in self._rand_ranges(T, self.t_num, self.t_len):
            out[:, s:e] *= 0.0
            mask_idx[:,s:e] = 1
        # Freq-mask
        for s, e in self._rand_ranges(F, self.f_num, self.f_len):
            out[s:e, :] *= 0.0
            mask_idx[s:e] = 1
        return out, mask_idx

    # -------- 호출부 --------
    def __call__(self, mag: torch.Tensor, is_idx: bool = False) -> torch.Tensor:
        """
        mag : (B, F, T) or (F, T)   torch.Tensor
        """
        if mag.ndim == 2:                      # (F,T) → (1,F,T)
            mag = mag.unsqueeze(0)

        B = mag.shape[0]
        if self.per_example:
            masked = torch.empty_like(mag)
            m_idx = torch.zeros_like(mag)
            for b in range(B):
                masked[b], m_idx[b] = self._mask_single(mag[b], m_idx[b])
            if is_idx:
                return masked, m_idx
            else:
                return masked
        else:
            mask0 = self._mask_single(mag[0])
            return mask0.unsqueeze(0).repeat(B, 1, 1)


# ── Step 1.1: BestModelTracker ───────────────────────────────────────────────

class BestModelTracker:
    """Track the best validation metric across epochs.

    Persisted as a lightweight JSON file alongside checkpoints so that
    training resume can restore the best-so-far state.

    Args:
        mode: 'min' for losses (lower is better), 'max' for metrics like PESQ.
        initial_best: Starting threshold. If None, the first update always wins.
    """

    _FILENAME = "best_tracker.json"

    def __init__(self, mode: str = "min", initial_best: float | None = None) -> None:
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got '{mode}'"
        self.mode = mode
        self.best_metric = initial_best
        self.best_epoch = -1

    def update(self, epoch: int, metric: float) -> bool:
        """Return True if *metric* is a new best."""
        if self.best_metric is None:
            is_best = True
        elif self.mode == "min":
            is_best = metric < self.best_metric
        else:
            is_best = metric > self.best_metric

        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
        return is_best

    def state_dict(self) -> dict:
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_metric = state["best_metric"]
        self.best_epoch = state["best_epoch"]
        self.mode = state.get("mode", self.mode)

    def save(self, directory: str) -> None:
        """Persist tracker state to best_tracker.json using atomic write."""
        path = os.path.join(directory, self._FILENAME)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.state_dict(), f)
        os.replace(tmp_path, path)

    def restore(self, directory: str) -> bool:
        """Restore from *directory*. Returns True on success, False if missing."""
        path = os.path.join(directory, self._FILENAME)
        if not os.path.exists(path):
            return False
        with open(path) as f:
            self.load_state_dict(json.load(f))
        logger.info(
            f"Restored BestModelTracker: best_epoch={self.best_epoch}, "
            f"best_metric={self.best_metric:.4e}"
        )
        return True


# ── Step 1.2: _enumerate_epoch_checkpoints ───────────────────────────────────

def _enumerate_epoch_checkpoints(checkpoint_dir: str) -> list[tuple[str, int]]:
    """Return all epoch checkpoint files in *checkpoint_dir* as (filepath, epoch) pairs.

    Only files matching the ``epoch.<NNNN>.(pt|pth|pkl)`` naming convention are
    returned. Non-checkpoint files in the directory are silently ignored.
    Results are sorted by epoch number in ascending order.

    Args:
        checkpoint_dir: Directory to scan.

    Returns:
        List of ``(filepath, epoch_number)`` tuples sorted by epoch ascending.
        Empty list if directory contains no matching files.
    """
    try:
        filenames = os.listdir(checkpoint_dir)
    except FileNotFoundError:
        return []
    result = []
    for fname in filenames:
        m = _EPOCH_PATTERN.match(fname)
        if m:
            epoch = int(m.group(1))
            result.append((os.path.join(checkpoint_dir, fname), epoch))
    result.sort(key=lambda x: x[1])
    return result


# ── Step 1.3: _prune_old_checkpoints ─────────────────────────────────────────

def _prune_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 5,
    keep_every_n: int = 10,
) -> None:
    """Keep the last *keep_last_n* epoch checkpoints plus every *keep_every_n*-th.

    Files named ``best_model.pth`` and ``best_tracker.json`` are never removed.
    Epoch 0 (``epoch.0000.pth``) is always kept because it seeds the adversarial
    fallback path.

    Args:
        checkpoint_dir: Directory containing epoch checkpoint files.
        keep_last_n: Number of most recent epoch checkpoints to keep (default 5).
        keep_every_n: Also keep checkpoints at every N-th epoch (default 10).
                      Set to 0 or None to disable.
    """
    paired = _enumerate_epoch_checkpoints(checkpoint_dir)
    if len(paired) <= keep_last_n:
        return

    epochs = [ep for _, ep in paired]
    keep_epochs: set[int] = set()
    keep_epochs.add(0)  # protect seed checkpoint (adversarial fallback path)
    # Keep the last keep_last_n epochs
    for _, ep in paired[-keep_last_n:]:
        keep_epochs.add(ep)
    # Keep every keep_every_n-th epoch
    if keep_every_n:
        for _, ep in paired:
            if ep % keep_every_n == 0:
                keep_epochs.add(ep)

    to_delete = [(path, ep) for path, ep in paired if ep not in keep_epochs]
    if to_delete:
        logger.info(
            f"Pruning {len(to_delete)} old checkpoints "
            f"(keeping last {keep_last_n} + every {keep_every_n}th + epoch 0)"
        )
    for path, ep in to_delete:
        os.remove(path)
        logger.debug(f"Pruned old checkpoint: {os.path.basename(path)}")


# ── Step 1.4: save_checkpoint_optimized ──────────────────────────────────────

def save_checkpoint_optimized(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    val_metric: float,
    best_tracker: "BestModelTracker",
    keep_last_n: int = 5,
    train_loss: float | None = None,
    valid_loss: float | None = None,
    rank: int = 0,
) -> bool:
    """Save training checkpoint, update best model, and prune old files.

    Replaces ``save_checkpoint_per_nth`` + ``save_checkpoint_per_best``.

    Write order guarantees crash safety:
    1. Save new checkpoint via atomic temp-write + fsync + os.replace.
    2. Copy best_model.pth atomically if metric improved.
    3. Prune old checkpoints last (no data loss if process dies mid-prune).

    DDP safety: only rank 0 performs writes. Other ranks return False immediately.

    Args:
        epoch: Current epoch number.
        model: Model whose state is saved.
        optimizer: Optimizer whose state is saved.
        checkpoint_path: Directory for saving checkpoints.
        val_metric: Scalar validation metric for best-model comparison.
        best_tracker: A ``BestModelTracker`` instance.
        keep_last_n: Number of recent epoch checkpoints to keep (default 5).
        train_loss: Optional training loss to embed in checkpoint dict.
        valid_loss: Optional validation loss to embed in checkpoint dict.
        rank: DDP process rank. Only rank 0 writes files (default 0).

    Returns:
        bool: True if this epoch is a new best.
    """
    if rank != 0:
        return False

    # 1. Build checkpoint dict (backward-compatible with existing format)
    ckpt_dict: dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if train_loss is not None:
        ckpt_dict["train_loss"] = train_loss
    if valid_loss is not None:
        ckpt_dict["valid_loss"] = valid_loss

    # Atomic save: tmp → fsync → rename
    final_path = os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth")
    tmp_path = os.path.join(checkpoint_path, f".epoch.{epoch:04}.pth.tmp")
    torch.save(ckpt_dict, tmp_path)
    fd = os.open(tmp_path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    os.replace(tmp_path, final_path)

    # 2. Best model tracking
    is_best = best_tracker.update(epoch, val_metric)
    if is_best:
        tmp_best = os.path.join(checkpoint_path, ".best_model.pth.tmp")
        shutil.copy2(final_path, tmp_best)
        os.replace(tmp_best, os.path.join(checkpoint_path, "best_model.pth"))
        logger.info(f"New best model at epoch {epoch} (metric={val_metric:.4e})")
    best_tracker.save(checkpoint_path)

    # 3. Prune old epoch checkpoints (always last)
    _prune_old_checkpoints(checkpoint_path, keep_last_n=keep_last_n)

    return is_best


def resolve_log_base(train_phase: str, config_name: str, base_dir: str) -> str:
    """Return the log directory base path, with backward-compatible fallback.

    Tries the new naming convention first (e.g. ``log_adversarial_baseline.yaml``),
    then falls back to the legacy ``_to48k`` naming (e.g.
    ``log_adversarial_to48k_baseline.yaml``) for checkpoints created before
    the config simplification.

    Args:
        train_phase: Current train phase name (e.g. ``"pretrain"``, ``"adversarial"``).
        config_name: Config file name (e.g. ``"baseline.yaml"``).
        base_dir: Root directory containing the ``log/`` folder.

    Returns:
        Relative log base path like ``log/log_{phase}_{config_name}``
        (new or legacy, whichever exists on disk; new name preferred).
    """
    new_base = f"log/log_{train_phase}_{config_name}"
    if os.path.isdir(os.path.join(base_dir, new_base, "weights")):
        return new_base
    # backward compat: try legacy _to48k naming
    legacy_base = f"log/log_{train_phase}_to48k_{config_name}"
    if os.path.isdir(os.path.join(base_dir, legacy_base, "weights")):
        return legacy_base
    # Neither exists yet — use new naming (will be created by makedirs)
    return new_base


# ── Step 1.5: setup_logging ───────────────────────────────────────────────────

def setup_logging(
    config_name: str,
    train_phase: str,
    base_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    config: dict,
    fs_src: int,
) -> tuple[str, "TBWriter", int]:
    """Create log directories, load latest checkpoint, and return a TBWriter.

    Constructs log paths via ``resolve_log_base()``, which tries the new naming
    convention first (e.g. ``log_pretrain_baseline.yaml``) then falls back to the
    legacy ``_to48k`` naming (e.g. ``log_pretrain_to48k_baseline.yaml``) for
    checkpoints created before the config simplification:
        ``<base_dir>/log/log_<train_phase>_<config_name>/weights``
        ``<base_dir>/log/log_<train_phase>_<config_name>/tensorboard``

    IMPORTANT — in-place mutation contract:
        ``model`` and ``optimizer`` are mutated in-place by checkpoint loading.
        Callers that reload after ``setup_logging()`` (e.g., adversarial fallback)
        MUST pass the SAME object references. Do not create fresh model/optimizer
        objects after this call.

    Note on config_name:
        The full config filename is kept as-is (e.g., ``baseline.yaml``) to match
        existing log directory structure. Stripping the extension would create new
        directories and break checkpoint continuity for ongoing training runs.

    Args:
        config_name: Config filename (e.g., ``baseline.yaml``). Kept with extension.
        train_phase: Training phase string (e.g., ``pretrain``, ``adversarial``).
        base_dir: Absolute path to the directory containing the ``log/`` folder
                  (typically ``os.path.dirname(os.path.abspath(__file__))``).
        model: Model to load checkpoint weights into (mutated in-place).
        optimizer: Optimizer to load checkpoint state into (mutated in-place).
        device: Device for loading the checkpoint (e.g., ``'cuda'``).
        config: Full YAML config dict (used to compute n_fft / n_hop for TBWriter).
        fs_src: Source sample rate in Hz.

    Returns:
        Tuple of ``(chkp_path, writer, start_epoch)``.
    """
    from tf_restormer.utils.util_writer import TBWriter

    log_base = resolve_log_base(train_phase, config_name, base_dir)
    chkp_path = os.path.join(base_dir, log_base, "weights")
    audio_log_path = os.path.join(base_dir, log_base, "tensorboard")
    os.makedirs(chkp_path, exist_ok=True)
    os.makedirs(audio_log_path, exist_ok=True)

    start_epoch = load_last_checkpoint_n_get_epoch(
        chkp_path, model, optimizer, location=device
    )

    n_fft = int(config["stft"]["frame_length"] * fs_src / 1000)
    n_hop = int(config["stft"]["frame_shift"] * fs_src / 1000)
    writer = TBWriter(logdir=audio_log_path, n_fft=n_fft, n_hop=n_hop, sr=fs_src)

    return chkp_path, writer, start_epoch


# ── Step 1.6: setup_optimizer_and_scheduler ──────────────────────────────────

def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict,
) -> tuple[
    torch.optim.Optimizer,
    "WarmupConstantSchedule",
    torch.optim.lr_scheduler.LRScheduler,
    type,
    type,
]:
    """Create optimizer, warmup scheduler, and main scheduler from config.

    Returns a 5-tuple so that the discriminator block in ``engine.py`` can reuse
    ``optim_cls`` / ``sched_cls`` without re-computing them via ``getattr``.

    SR_CorrNet difference: SR_CorrNet returns a 3-tuple. The extra two class
    references are a TF_Restormer extension required by the adversarial stage
    discriminator optimizer, which reuses the same class types with different
    config keys (``optimizer_D``).

    Args:
        model: Model whose parameters are passed to the optimizer.
        config: Full YAML config dict. Expected keys under ``config["engine"]``:
            - ``optimizer.name``: e.g. ``"Adam"``
            - ``optimizer.<name>``: kwargs dict (e.g. ``Adam: {lr: 1e-3}``)
            - ``scheduler.name``: e.g. ``"CosineAnnealingLR"``
            - ``scheduler.WarmupConstantSchedule``: warmup kwargs
            - ``scheduler.<name>``: kwargs dict for the main scheduler

    Returns:
        ``(optimizer, warmup_scheduler, main_scheduler, optim_cls, sched_cls)``
    """
    engine_cfg = config["engine"]
    optim_cls = getattr(torch.optim, engine_cfg["optimizer"]["name"])
    sched_cls = getattr(torch.optim.lr_scheduler, engine_cfg["scheduler"]["name"])

    optimizer = optim_cls(
        model.parameters(),
        **engine_cfg["optimizer"].get(engine_cfg["optimizer"]["name"], {}),
    )
    warmup_scheduler = WarmupConstantSchedule(
        optimizer,
        **engine_cfg["scheduler"]["WarmupConstantSchedule"],
    )
    main_scheduler = sched_cls(
        optimizer,
        **engine_cfg["scheduler"].get(engine_cfg["scheduler"]["name"], {}),
    )
    return optimizer, warmup_scheduler, main_scheduler, optim_cls, sched_cls


# ── Step 1.7: Logging utility constants and functions ────────────────────────

PBAR_FMT = (
    "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)

# Keys whose values are on a perceptual / dB-like scale — display as .3f
# Note: TF_Restormer's training loss keys (L_se, L_time, L_rep, etc.) do not
# appear in this set, so all training losses use .2e format. This set is
# retained for potential use in engine_eval.py where evaluation metrics
# (PESQ, STOI, SDR) benefit from fixed-point formatting.
_LOG_SCALE_KEYS: set[str] = {"PESQ", "STOI", "SDR"}


def format_pbar(dict_loss: dict) -> dict:
    """Format a loss dict for tqdm ``set_postfix``.

    Values whose key is in ``_LOG_SCALE_KEYS`` are formatted as ``.3f``
    (perceptual metrics); all others use ``.2e`` (loss values).

    Args:
        dict_loss: Dict of {metric_name: float_value}.

    Returns:
        Dict of {metric_name: formatted_string}.
    """
    return {
        k: (f"{v:.3f}" if k in _LOG_SCALE_KEYS else f"{v:.2e}")
        for k, v in dict_loss.items()
    }


def log_scalars_to_tb(
    writer: "TBWriter",
    metric_dict: dict,
    epoch: int,
) -> None:
    """Log each key in *metric_dict* as an individual TensorBoard scalar.

    Keys should use the ``'group/partition'`` format (e.g., ``'Loss_se/train'``,
    ``'PESQ/valid'``) so TensorBoard auto-groups related charts.

    Note: This uses ``add_scalar`` (singular), not ``add_scalars`` (plural).
    This produces cleaner per-chart grouping compared to the grouped dict format
    used by ``add_scalars``.

    Args:
        writer: A ``TBWriter`` (or ``SummaryWriter``) instance.
        metric_dict: Dict of {tag: scalar_value}.
        epoch: Global step / epoch number for the x-axis.
    """
    for tag, value in metric_dict.items():
        writer.add_scalar(tag, value, epoch)
    writer.flush()


def format_epoch_log(
    config_name: str,
    epoch: int,
    stage: str,
    metrics: dict,
    elapsed_sec: float,
    suffix: str = "",
) -> str:
    """Build a single-line epoch log string.

    Args:
        config_name: Config filename for identification (e.g., ``baseline.yaml``).
        epoch: Current epoch number.
        stage: Stage label, e.g. ``'TRAIN'``, ``'VALID'``, ``'INIT'``.
        metrics: Dict of ``{key: float_value}`` to display.
        elapsed_sec: Wall-clock seconds elapsed for this stage.
        suffix: Optional trailing text appended after the timing info,
                e.g. ``' (improved)'`` or ``'(256 batches)'``.

    Returns:
        Formatted log string.
    """
    header = f"[{config_name}]\n[Epoch {epoch}] {stage} ({elapsed_sec:.1f}s){suffix}"
    metric_parts = []
    for k, v in metrics.items():
        if k in _LOG_SCALE_KEYS:
            metric_parts.append(f"{k}={v:.3f}")
        else:
            metric_parts.append(f"{k}={v:.2e}")
    metric_str = " | ".join(metric_parts)
    return f"{header} {metric_str}" if metric_str else header


# ── Step 1.8: load_last_checkpoint_n_get_epoch_model_only ────────────────────

def load_last_checkpoint_n_get_epoch_model_only(
    checkpoint_dir: str,
    model: torch.nn.Module,
    location: torch.device | str = "cuda",
) -> None:
    """Load the latest checkpoint (model weights only) from *checkpoint_dir*.

    A cleaner alternative to calling ``load_last_checkpoint_n_get_epoch`` with
    ``optimizer=None``. Intended for eval and inference paths where optimizer
    state is never needed.

    Primary search: files matching ``epoch.*.pth`` naming convention.
    No fallback to arbitrary ``.pt`` / ``.pth`` files (unlike
    ``_find_latest_checkpoint``).

    Args:
        checkpoint_dir: Directory containing ``epoch.*.pth`` files.
        model: Model to load weights into (mutated in-place).
        location: Device for loading the checkpoint (default ``'cuda'``).

    Returns:
        None. If no checkpoint is found, returns without modifying the model.
    """
    # Prefer best_model.pth when available — it contains the highest-quality weights.
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        chosen_path = best_model_path
    else:
        pairs = _enumerate_epoch_checkpoints(checkpoint_dir)
        if not pairs:
            logger.warning(
                f"No checkpoint found in {checkpoint_dir}, using random weights"
            )
            return
        chosen_path, _ = pairs[-1]

    logger.info(f"Loaded model weights from {chosen_path}")
    checkpoint_dict = torch.load(chosen_path, map_location=location, weights_only=True)
    model_state = checkpoint_dict["model_state_dict"]
    model_state = _fix_compiled_state_dict(model_state)
    model.load_state_dict(model_state, strict=False)

