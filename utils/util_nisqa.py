# -*- coding: utf-8 -*-
"""
Simplified NISQA wrapper for direct audio input
Based on original NISQA by Gabriel Mittag, TU-Berlin
"""
import os
import sys
import torch
import librosa
import numpy as np
from pathlib import Path

class NISQAModel:
    """
    Simplified NISQA model wrapper for MOS prediction from audio signal
    """
    def __init__(self, model_path=None, pretrained_model=None, device='cuda'):
        """
        Initialize NISQA model
        Args:
            model_path: Path to NISQA_models directory
            pretrained_model: Path to pretrained model .tar file
        """
        if model_path is None:
            current_dir = Path(__file__).parent
            model_path = current_dir / "NISQA_models"
        
        self.model_path = Path(model_path)
        self.device = device        
        # Default model parameters from NISQA (exact values from nisqa_mos_only.tar)
        self.args = {
            'ms_seg_length': 15,
            'ms_n_mels': 48,
            'ms_n_fft': 4096,         # Exact value from checkpoint
            'ms_hop_length': 0.01,    # 10ms in seconds (will be converted to samples)
            'ms_win_length': 0.02,    # 20ms in seconds (will be converted to samples)
            'ms_sr': None,            # None = use original sample rate (NO resampling!)
            'ms_fmax': 20000,         # 20kHz (exact value from checkpoint)
            'ms_channel': None,
        }
        
        # Add NISQA_models to path
        sys.path.insert(0, str(self.model_path))
        
        # Load model
        self._load_model(pretrained_model)
    
    def _load_model(self, pretrained_model=None):
        """Load NISQA model and weights"""
        from NISQA_lib import NISQA
        
        # Load pretrained model if provided
        if pretrained_model is None:
            pretrained_model = self.model_path / "nisqa_mos_only.tar"
        
        if not Path(pretrained_model).exists():
            raise FileNotFoundError(f"Model file not found: {pretrained_model}")
        
        checkpoint = torch.load(pretrained_model, map_location=self.device)
        
        # Get model arguments from checkpoint['args'] (not 'model_args'!)
        if 'args' in checkpoint:
            # Checkpoint uses 'args' field with all parameters
            args = checkpoint['args']
            
            # Extract only NISQA model parameters (not training parameters)
            model_args = {
                'ms_seg_length': args['ms_seg_length'],
                'ms_n_mels': args['ms_n_mels'],
                'cnn_model': args['cnn_model'],
                'cnn_c_out_1': args['cnn_c_out_1'],
                'cnn_c_out_2': args['cnn_c_out_2'],
                'cnn_c_out_3': args['cnn_c_out_3'],
                'cnn_kernel_size': args['cnn_kernel_size'],  # (3, 3) tuple!
                'cnn_dropout': args['cnn_dropout'],
                'cnn_pool_1': args['cnn_pool_1'],
                'cnn_pool_2': args['cnn_pool_2'],
                'cnn_pool_3': args['cnn_pool_3'],
                'cnn_fc_out_h': args['cnn_fc_out_h'],
                'td': args['td'],
                'td_sa_d_model': args['td_sa_d_model'],
                'td_sa_nhead': args['td_sa_nhead'],
                'td_sa_pos_enc': args['td_sa_pos_enc'],  # False, not None!
                'td_sa_num_layers': args['td_sa_num_layers'],
                'td_sa_h': args['td_sa_h'],
                'td_sa_dropout': args['td_sa_dropout'],
                'td_lstm_h': args['td_lstm_h'],
                'td_lstm_num_layers': args['td_lstm_num_layers'],
                'td_lstm_dropout': args['td_lstm_dropout'],
                'td_lstm_bidirectional': args['td_lstm_bidirectional'],
                'td_2': args['td_2'],
                'td_2_sa_d_model': args['td_2_sa_d_model'],
                'td_2_sa_nhead': args['td_2_sa_nhead'],
                'td_2_sa_pos_enc': args['td_2_sa_pos_enc'],
                'td_2_sa_num_layers': args['td_2_sa_num_layers'],
                'td_2_sa_h': args['td_2_sa_h'],
                'td_2_sa_dropout': args['td_2_sa_dropout'],
                'td_2_lstm_h': args['td_2_lstm_h'],
                'td_2_lstm_num_layers': args['td_2_lstm_num_layers'],
                'td_2_lstm_dropout': args['td_2_lstm_dropout'],
                'td_2_lstm_bidirectional': args['td_2_lstm_bidirectional'],
                'pool': args['pool'],
                'pool_att_h': args['pool_att_h'],
                'pool_att_dropout': args['pool_att_dropout'],  # 0, not 0.1!
            }
        else:
            raise KeyError("Checkpoint does not contain 'args' field!")
        
        # Initialize model
        self.model = NISQA(**model_args)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"NISQA 모델 로드 완료: {self.device}")
    
    def _audio_to_mel_spec(self, x, fs):
        """
        Convert audio signal to mel-spectrogram segments
        Args:
            x: Audio signal (numpy array or torch tensor)
            fs: Sample rate
        Returns:
            mel_spec: Mel-spectrogram tensor [1, n_wins, 1, n_mels, seg_length]
            n_wins: Number of segments
        """
        # Convert to numpy if tensor
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        # Ensure mono
        if x.ndim > 1:
            x = np.mean(x, axis=0)
        
        # NO resampling! NISQA uses original sample rate (ms_sr: None in checkpoint)
        # Convert hop_length and win_length from seconds to samples (same as original NISQA)
        hop_length = int(fs * self.args['ms_hop_length'])
        win_length = int(fs * self.args['ms_win_length'])
        
        # Calculate mel-spectrogram (amplitude spectrum, not power!)
        mel_spec = librosa.feature.melspectrogram(
            y=x,
            sr=fs,
            n_mels=self.args['ms_n_mels'],
            n_fft=self.args['ms_n_fft'],
            hop_length=hop_length,
            win_length=win_length,
            fmax=self.args['ms_fmax'],
            power=1.0,  # Amplitude spectrum (same as original NISQA)
            window='hann',
            center=True,
            pad_mode='reflect',
            fmin=0.0,
            htk=False,
            norm='slaney',
        )
        
        # Convert to dB scale (same as original NISQA)
        mel_spec = librosa.core.amplitude_to_db(
            mel_spec, 
            ref=1.0,      # NOT ref=np.max!
            amin=1e-4, 
            top_db=80.0
        )
        
        # NO normalization! Original NISQA uses [-80, 0] dB range directly
        
        # Segment using NISQA's segment_specs logic (NOT simple chunking!)
        # This creates overlapping windows with seg_length=15
        seg_length = self.args['ms_seg_length']
        
        if seg_length % 2 == 0:
            raise ValueError(f'seg_length must be odd! (seg_length={seg_length})')
        
        # Convert to torch tensor
        if not torch.is_tensor(mel_spec):
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        
        # Calculate number of segments
        n_wins = mel_spec.shape[1] - (seg_length - 1)
        if n_wins < 1:
            raise ValueError(
                f"Audio too short. Only {mel_spec.shape[1]} frames but seg_length={seg_length}. "
                f"Need at least {seg_length} frames."
            )
        
        # Create overlapping segments using broadcast indexing
        # This is the "broadcast magic" from segment_specs function
        idx1 = torch.arange(seg_length)  # [0, 1, 2, ..., 14]
        idx2 = torch.arange(n_wins)      # [0, 1, 2, ..., n_wins-1]
        idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)  # [n_wins, seg_length] index matrix
        
        # Apply indexing: [n_mels, n_frames] -> [n_wins, seg_length, n_mels]
        mel_spec = mel_spec.transpose(1, 0)[idx3, :].unsqueeze(1).transpose(3, 2)
        # Result shape: [n_wins, 1, n_mels, seg_length]
        
        # Apply seg_hop (downsampling of segments)
        # Default seg_hop=4 from checkpoint
        seg_hop = 4  # From checkpoint args
        if seg_hop > 1:
            mel_spec = mel_spec[::seg_hop, :]
            n_wins = int(np.ceil(n_wins / seg_hop))
        
        # Add batch dimension: [1, n_wins, 1, n_mels, seg_length]
        mel_spec = mel_spec.unsqueeze(0)
        n_wins = torch.LongTensor([n_wins])
        
        return mel_spec, n_wins
    
    def forward(self, x, fs):
        """
        Compute NISQA MOS score
        Args:
            x: Audio signal (numpy array or torch tensor)
            fs: Sample rate
        Returns:
            mos: MOS score (raw output from model)
        """
        if self.model is None:
            raise RuntimeError("NISQA model not loaded")
        
        # Convert audio to mel-spectrogram segments
        mel_spec, n_wins = self._audio_to_mel_spec(x, fs)
        mel_spec = mel_spec.to(self.device)
        n_wins = n_wins.to(self.device)
        
        # Debug: print input shapes
        # print(f"[NISQA Debug] mel_spec shape: {mel_spec.shape}, n_wins: {n_wins}")
        # print(f"[NISQA Debug] mel_spec stats - min: {mel_spec.min():.4f}, max: {mel_spec.max():.4f}, mean: {mel_spec.mean():.4f}")
        
        # Model inference
        with torch.no_grad():
            output = self.model(mel_spec, n_wins)
            # print(f"[NISQA Debug] Raw model output: {output}")
            # print(f"[NISQA Debug] Output shape: {output.shape}, dtype: {output.dtype}")
            
            mos = float(output.cpu().item())
            # print(f"[NISQA Debug] Final MOS (no clamping): {mos}")
            
            return mos
