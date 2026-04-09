# Copyright 2024 Takaaki Saeki
# MIT LICENSE (https://opensource.org/license/mit/)
# Adapted for TF_Restormer project - Using pesq package instead of pypesq

from typing import Dict, List, Tuple
import numpy as np
import librosa
from pesq import pesq as calculate_pesq


class PESQ:
    def __init__(self, sr=16000):
        """
        Args:
            sr (int): Sampling rate.
        """
        self.sr = sr
        self.tar_fs = 16000
    
    def score(self, gt_wav, gen_wav):
        """
        Calculate PESQ score using the pesq package.
        
        Args:
            gt_wav (np.ndarray): Ground truth waveform (T,).
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: PESQ score.
        """
        # Resample to 16kHz if necessary
        if self.sr != self.tar_fs:
            gt_wav = librosa.resample(gt_wav.astype(np.float32), orig_sr=self.sr, target_sr=self.tar_fs)
            gen_wav = librosa.resample(gen_wav.astype(np.float32), orig_sr=self.sr, target_sr=self.tar_fs)
        
        # Determine mode based on sample rate
        mode = 'wb' if self.tar_fs == 16000 else 'nb'
        
        # Calculate PESQ score
        score = calculate_pesq(self.tar_fs, gt_wav, gen_wav, mode)
        return score