import os
import torch
import numpy as np
from loguru import logger

from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import torch
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import zscore
import pandas as pd
from scipy.spatial.distance import cosine
import random



class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            return 1.0
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)






def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
    """
    Load the latest checkpoint (model state and optimizer state) from a given directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model into which the checkpoint's model state should be loaded.
        optimizer (torch.optim.Optimizer): The optimizer into which the checkpoint's optimizer state should be loaded.
        location (str, optional): Device location for loading the checkpoint. Defaults to 'cpu'.

    Returns:
        int: The epoch number associated with the loaded checkpoint. 
             If no checkpoint is found, returns 0 as the starting epoch.

    Notes:
        - The checkpoint file is expected to have keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'.
        - If there are multiple checkpoint files in the directory, the one with the highest epoch number is loaded.
    """
    # List all .pkl files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]

    # If there are no checkpoint files, return 0 as the starting epoch
    if not checkpoint_files: return 1
    else:
        # Extract the epoch numbers from the file names and find the latest (max)
        epochs = [int(f.split('.')[1]) for f in checkpoint_files]
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[epochs.index(max(epochs))])


        # Load the checkpoint into the model & optimizer
        logger.info(f"Loaded Pretrained model from {latest_checkpoint_file} .....")
        checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location)
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False) # Depend on weight file's key!!
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        
        # Retrun latent epoch
        return checkpoint_dict['epoch'] + 1
    

# def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
#     import torch
#     import os

#     checkpoint_files = [f for f in os.listdir(checkpoint_dir)]
#     if not checkpoint_files:
#         return 1
#     else:
#         epochs = [int(f.split('.')[1]) for f in checkpoint_files]
#         latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[epochs.index(max(epochs))])

#         logger.info(f"Loaded Pretrained model from {latest_checkpoint_file} .....")
#         checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location)
#         pretrained_dict = checkpoint_dict['model_state_dict']
#         model_dict = model.state_dict()

#         # Shape 체크 후 맞는 것만 불러오기
#         filtered_dict = {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict and v.shape == model_dict[k].shape:
#                 filtered_dict[k] = v
#             else:
#                 print(f"Skipped loading parameter: {k}, shape mismatch {v.shape} vs {model_dict.get(k, 'Not found').shape}")

#         model.load_state_dict(filtered_dict, strict=False)
#         optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

#         return checkpoint_dict['epoch'] + 1

# def load_average_last_checkpoints(checkpoint_dir, model, optimizer, location, num_checkpoints=3):
#     """
#     Load the last `num_checkpoints` checkpoints, compute the average of their weights, 
#     and update the model's state dictionary with the averaged weights.

#     Args:
#         checkpoint_dir (str): Directory containing the checkpoint files.
#         model (torch.nn.Module): The model to which the averaged state should be loaded.
#         optimizer (torch.optim.Optimizer): The optimizer for compatibility (not updated in this function).
#         location (str): Device location for loading the checkpoints. Defaults to 'cpu'.
#         num_checkpoints (int): Number of most recent checkpoints to average. Defaults to 5.

#     Returns:
#         int: The epoch number of the most recent checkpoint.

#     Notes:
#         - Checkpoints should have keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'.
#         - Assumes checkpoint file names include epoch numbers in the format '...epoch.NUMBER...'.
#     """
#     # List all checkpoint files in the directory
#     checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

#     if not checkpoint_files:
#         raise ValueError("No checkpoint files found in the specified directory.")

#     # Extract epoch numbers and sort files by epoch number in descending order
#     checkpoints = [(int(f.split('.')[1]), f) for f in checkpoint_files]
#     checkpoints = sorted(checkpoints, key=lambda x: x[0], reverse=True)

#     # Get the latest `num_checkpoints` files
#     selected_checkpoints = checkpoints[:num_checkpoints]

#     # Initialize a dictionary to accumulate weights
#     averaged_state_dict = None
#     total_checkpoints = len(selected_checkpoints)

#     for epoch, file_name in selected_checkpoints:
#         checkpoint_path = os.path.join(checkpoint_dir, file_name)
#         checkpoint = torch.load(checkpoint_path, map_location=location)
#         model_state_dict = checkpoint['model_state_dict']

#         if averaged_state_dict is None:
#             # Initialize averaged_state_dict with the first checkpoint
#             averaged_state_dict = {key: torch.zeros_like(value) for key, value in model_state_dict.items()}

#         # Accumulate weights
#         for key in averaged_state_dict:
#             averaged_state_dict[key] += model_state_dict[key] / total_checkpoints

#     # Load the averaged weights into the model
#     model.load_state_dict(averaged_state_dict, strict=False)

#     # Return the most recent epoch number
#     return selected_checkpoints[0][0] + 1



def save_checkpoint_per_nth(nth, epoch, model, optimizer, train_loss, valid_loss, checkpoint_path):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

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

def save_checkpoint_per_best(best, valid_loss, train_loss, epoch, model, optimizer, checkpoint_path):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if valid_loss < best:
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
        
        best = valid_loss
    return best

def step_scheduler(scheduler, **kwargs):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    # Add another schedulers
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler)}")

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_parameters += param_count
        logger.info(f"{name}: {param_count}")
    logger.info(f"Total parameters: {(total_parameters / 1e6):.2f}M")

def model_params_mac_summary(model, input_shape, metrics, device):
    
    # ptflpos
    if 'ptflops' in metrics:
        MACs_ptflops, params_ptflops = get_model_complexity_info(model, (*input_shape,), print_per_layer_stat=False, verbose=False) # (num_samples,)
        MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
        logger.info(f"ptflops: MACs: {MACs_ptflops}, Params: {params_ptflops}")

    # thop
    if 'thop' in metrics:
        input = torch.randn(1, *input_shape).to(device)
        MACs_thop, params_thop = profile(model, inputs=input, verbose=False)
        MACs_thop, params_thop = MACs_thop/1e9, params_thop/1e6
        logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")
    
    # torchinfo
    if 'torchinfo' in metrics:
        model_profile = summary_(model, input_size=(1, *input_shape), verbose=0, device=device)
        MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e9, model_profile.total_params/1e6
        logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")


    # MEASURE PERFORMANCE
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 500
    # repetitions2 = 500
    # timings = np.zeros((repetitions,1))
    # torch.set_num_threads(1)
    # with torch.no_grad():
    #     for rep in range(repetitions+repetitions2):
    #         if rep > repetitions:
    #             starter.record()
    #             _ = model(dummy_input)
    #             ender.record()
    #             # WAIT FOR GPU SYNC
    #             torch.cuda.synchronize()
    #             curr_time = starter.elapsed_time(ender)
    #             timings[rep-repetitions2] = curr_time
    # logger.info(f"Timing: {timings.mean()}")


def create_sampler(dataset_size, num_samples):
    indices = list(range(dataset_size))
    random.shuffle(indices)
    sampled_indices = indices[:num_samples]
    return torch.utils.data.SubsetRandomSampler(sampled_indices)



def p_law_compress(x, c=0.3, mode=None):
    # mode : None / inverse
    x_abs = torch.clamp(x.abs(), min=1.0e-15)
    x_angle = x.angle()
    x_compressed = x_abs**(1/c) if mode == 'inverse' else x_abs**c
    return torch.polar(x_compressed, x_angle)


class RandSpecAugment:
    """
    PyTorch 기반 SpecAugment (배치별 개별 마스크 지원)

    파라미터는 전부 [min, max] 범위로 지정
      • time-mask 길이   : t_len
      • freq-mask 길이   : f_len
      • time-mask 개수  : t_num
      • freq-mask 개수  : f_num
    per_example=True  →  배치의 각 샘플에 독립 마스크
    """
    def __init__(self,
                 t_len=(10, 20), f_len=(40, 60),
                 t_num=(2,  5),  f_num=(1, 2),
                 per_example=True, device="cpu"):
        self.t_len, self.f_len = t_len, f_len
        self.t_num, self.f_num = t_num, f_num
        self.per_example = per_example
        self.device = device

    # -------- 겹치지 않는 1-D 구간 뽑기 (pure python) --------
    @staticmethod
    def _rand_ranges(L: int, n_rng, w_rng):
        n = random.randint(*n_rng)
        w_min, w_max = w_rng
        ranges = []
        for _ in range(n):
            for _ in range(10):                     # 10회까지 재시도
                w = random.randint(w_min, w_max)
                if w >= L:
                    continue
                s = random.randint(0, L - w)
                if all(not (s < e and s+w > b) for b, e in ranges):
                    ranges.append((s, s+w))
                    break
        return ranges

    # -------- 단일 샘플 마스킹 --------
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
    def __call__(self, mag: torch.Tensor, is_idx=False) -> torch.Tensor:
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
    
