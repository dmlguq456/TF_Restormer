import argparse
import os
import torch
import random
import librosa as audio_lib
import numpy as np
import scipy.signal as ss
from glob import glob
import pickle
from os.path import relpath
from tf_restormer.utils import util_dataset
from tf_restormer.utils.decorators import logger_wraps
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from pedalboard import LowpassFilter, HighpassFilter, Distortion , Clipping, MP3Compressor
from scipy.signal import filtfilt, firwin2
import colorednoise
from torchaudio.functional import resample as torch_resample


@logger_wraps()
def get_dataloaders(args: argparse.Namespace, dataset_config: dict, loader_config: dict) -> dict:
    """Create train/valid/test DataLoaders based on engine mode.

    Args:
        args: CLI arguments (engine_mode determines partition split).
        dataset_config: Dataset section of YAML config (flat, no phase sub-key).
        loader_config: DataLoader section of YAML config.

    Returns:
        Dict of DataLoaders keyed by partition name ('train', 'valid', 'test').
    """
    # create dataset object for each partition
    partitions = ["train", "valid"] if args.engine_mode == "train"  else ["test"]
    dataloaders = {}

    for partition in partitions:
        if partition in ["train", "valid"]:
            dataset = SynthesisDataset(partition, dataset_config, dataset_config['synthesis_config'])
        else:
            testset_name = dataset_config['testset_key']
            dataset = EvalDataset(dataset_config[testset_name])
            
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1 if partition == 'test' else loader_config["batch_size"],
            shuffle = False if partition == 'test' else True, # only train: (partition == 'train') / all: True
            pin_memory = loader_config["pin_memory"],
            num_workers = loader_config["num_workers"],
            drop_last = loader_config["drop_last"])
        dataloaders[partition] = dataloader
    return dataloaders


@logger_wraps()
class SynthesisDataset(Dataset):
    def __init__(self, partition: str, dataset_config: dict, synthesis_config: dict) -> None:
        """On-the-fly noisy signal synthesis dataset (clean + noise + RIR mixing, not TTS) for training and validation."""

        # load wave scp
        wave_scp_src = os.path.join(dataset_config["scp_dir"], dataset_config[partition]['spk'])
        wave_scp_noise = os.path.join(dataset_config["scp_dir"], dataset_config[partition]['noise'])
        # Get DB_ROOT from config or environment
        db_root = dataset_config.get('db_root', None)
        
        self.wave_dict_src = util_dataset.parse_scps(wave_scp_src, db_root)
        self.wave_keys = list(self.wave_dict_src.keys())
        self.wave_dict_noise = util_dataset.parse_scps(wave_scp_noise, db_root)
        self.wave_noise_keys = list(self.wave_dict_noise.keys())

        # RIR directory from YAML config (required)
        rir_dir = dataset_config.get('rir_dir')
        if not rir_dir:
            raise ValueError("'rir_dir' not set in YAML config. Set it to your RIR directory path.")
        
        if not os.path.exists(rir_dir):
            raise ValueError(f"RIR directory does not exist: {rir_dir}")
        RIR_cache_path = os.path.join(rir_dir, 'rir_list_cache.pkl')
        if os.path.isfile(RIR_cache_path):
            with open(RIR_cache_path, 'rb') as f:
                self.list_RIR = pickle.load(f)
        else:
            self.list_RIR = glob(os.path.join(rir_dir, '**', '*.wav'), recursive=True)
            with open(RIR_cache_path, 'wb') as f: 
                pickle.dump(self.list_RIR, f)

        self.fs_src = dataset_config['sample_rate_src']
        self.fs = 16000 #! noise are loaded and truncated based on fs=16000 for efficiency
        self.fs_in = dataset_config['sample_rate_in']
        self.dur = dataset_config['max_len'] # seconds

        # ------ synthesis_config ----- #
        # ------- RIR ------ #
        self.rir_prob = synthesis_config['rir']['prob']
        self.rir_sidelobe = int(synthesis_config['rir']['rir_sidelobe'] * 0.001 * self.fs_src)
        # ------- NOISE ------ #
        self.colored_beta_range = synthesis_config['noise']['c_beta_range']
        self.SNR_range = synthesis_config['noise']['SNR_range']
        self.c_SNR_range = synthesis_config['noise']['c_SNR_range']
        # ------- BPF ------ #
        self.BPF_prob = synthesis_config['BPF']['prob']
        self.low_cutoff_range = synthesis_config['BPF']['low_cutoff_freq_range']
        self.BPF_beta_range = synthesis_config['BPF']['fir_filter_beta']
        # ------- clipping ------ #
        self.clipping_prob = synthesis_config['clipping']['prob']
        self.clipping_level_range = synthesis_config['clipping']['clipping_level_range']
        # ------- level ------- #
        self.target_dB_FS = synthesis_config['level']['target_dB_FS']
        
        # ------ Multi-speaker concat probability ------ #
        self.multi_spk_prob = synthesis_config.get('multi_spk_prob', 0.3)  # 30% chance to concat another speaker


    def occlusion_fir(self, 
                      audio,
                      numtaps_range=(31, 61),
                      slope_bound = (200,500),
                      gain_range=(0.1, 0.3)):
    
        f1 = np.random.randint(*self.low_cutoff_range)
        f2 = f1 + np.random.randint(*slope_bound)
        bands = [0, f1, f2, 8000]
        cut_gain = np.random.uniform(*gain_range)
        gains = [1, 1, cut_gain, cut_gain]
        beta=random.uniform(*self.BPF_beta_range)
        gains = [g**beta for g in gains]  # considering filtfilt

        numtaps = np.random.randint(numtaps_range[0], numtaps_range[1] + 1)
        if numtaps % 2 == 0: numtaps += 1

        fir = firwin2(numtaps, bands, gains, fs=self.fs)
        return filtfilt(fir, [1.0], audio)


    def _noise(self, clean, noise, c_noise, eps=1.0e-8):
        clean_rms = (clean ** 2).mean() ** 0.5
       
        noise_rms = (noise ** 2).mean() ** 0.5
        SNR = np.random.randint(*self.SNR_range)
        snr_scalar = clean_rms / (10 ** (SNR / 20)) / (noise_rms + eps)
        noise *= snr_scalar

        c_noise_rms = (c_noise ** 2).mean() ** 0.5
        c_SNR = np.random.randint(*self.c_SNR_range)
        c_snr_scalar = clean_rms / (10 ** (c_SNR / 20)) / (c_noise_rms + eps)
        c_noise *= c_snr_scalar
        return clean + noise + c_noise

    def _clipping(self, src):
        clipping_level = np.random.randint(*self.clipping_level_range)
        clipping = Clipping(clipping_level)
        srcs = clipping(src, self.fs)
        return srcs

    def _compress(self, src):
        compress_level = np.random.randint(*self.MP3_compress_range)
        compressor = MP3Compressor(compress_level)
        srcs = compressor(src, self.fs)
        return srcs

    def _resample(self, src):
        srcs_down = audio_lib.resample(src, orig_sr=self.fs, target_sr=self.fs_in)
        return srcs_down
    

    def _synthesis(self, clean, noise, rir):
        
        # reverberation
        rir_d = np.zeros(rir.shape)
        idx = util_dataset.find_peak(rir, self.fs_src)
        start, end = max(0, idx-self.rir_sidelobe//2), idx+self.rir_sidelobe//2+1
        rir_d[start:end] = rir[start:end]
        if random.random() < self.rir_prob:
            clean_rir = ss.fftconvolve(clean, rir)[:len(clean)]
            clean = ss.fftconvolve(clean, rir_d)[:len(clean)]
        else:
            clean_rir = clean
        # down_sampling to fs=16k preliminary for efficient computation
        if self.fs_src > self.fs:
            clean_rir = audio_lib.resample(clean_rir, orig_sr=self.fs_src, target_sr=self.fs)
        clean_rir, _, scalar = util_dataset.tailor_dB_FS(clean_rir)
        clean *= scalar

        # Generate colored gaussian noise
        c_noise = colorednoise.powerlaw_psd_gaussian(random.uniform(*self.colored_beta_range), int(self.fs*self.dur))

        # Low-Pass Fitering
        if self.BPF_prob > random.random():
            clean_rir = self.occlusion_fir(clean_rir)
            c_noise = self.occlusion_fir(c_noise)
            noise = self.occlusion_fir(noise)

        # Noise addition with SNR (interf & colored noise)
        noisy_distort = self._noise(clean_rir, noise, c_noise)

        # Distortion by Clipping and Compression
        if self.clipping_prob > random.random():
            noisy_distort = self._clipping(noisy_distort)
        
        # resampling
        # noisy_distort = self._resample(noisy_distort)

		# rescale noisy RMS
        noisy_target_dB_FS = np.random.randint(*self.target_dB_FS)
        noisy_distort, _, noisy_scalar = util_dataset.tailor_dB_FS(noisy_distort, noisy_target_dB_FS)
        clean *= noisy_scalar
        if np.any(np.abs(noisy_distort) > 0.999):
            noisy_scalar = np.max(np.abs(noisy_distort)) / 0.99  # same as divide by 1
            noisy_distort = noisy_distort / noisy_scalar
            clean = clean / noisy_scalar

        return noisy_distort, clean
	
    def _load(self, key):

        # load speech source
        files = self.wave_dict_src[key]
        file_dur = audio_lib.get_duration(filename=files)
        
        clean = audio_lib.load(files, sr=self.fs_src)[0]
        
        # Randomly decide if we want to add another speaker (multi-speaker scenario)
        if random.random() < self.multi_spk_prob:
            key_2 = random.choice(self.wave_keys)
            
            files_2 = self.wave_dict_src[key_2]
            clean_2 = audio_lib.load(files_2, sr=self.fs_src)[0]
            
            # Concatenate the two clean signals
            clean = np.concatenate([clean, clean_2])
        
        # load noise source randomly
        key_n = random.choice(self.wave_noise_keys)
        files_n = self.wave_dict_noise[key_n]
        file_dur = audio_lib.get_duration(filename=files_n)
        if file_dur > self.dur:
            max_offset = file_dur - self.dur
            offset = random.uniform(0, max_offset)
            noise = audio_lib.load(files_n, sr=self.fs, offset=offset, duration=self.dur)[0]
        else:
            noise = audio_lib.load(files_n,sr=self.fs)[0]

        # load RIR randomly
        path_RIR = random.choice(self.list_RIR)
        RIR = audio_lib.load(path_RIR,sr=self.fs_src, mono=False)[0]
        if len(RIR.shape) > 1:
            RIR = RIR[random.randint(0,RIR.shape[0]-1)]

        ## Length Match - clean is already processed above
        clean,_ = util_dataset.match_length(clean, int(self.fs_src*self.dur))
        noise,_ = util_dataset.match_length(noise, int(self.fs*self.dur))

        # mix 
        noisy_distort, clean = self._synthesis(clean, noise, RIR)

        return {"num_sample":noisy_distort.shape[0], 
                "noisy_distort":noisy_distort, 
                "clean":clean}


    def __getitem__(self, idx: int) -> dict:
        key = self.wave_keys[idx]
        return self._load(key)


    def __len__(self) -> int:
        return len(self.wave_dict_src)



class EvalDataset(Dataset):
    def __init__(self, dataset_config: dict) -> None:
        """File-based evaluation dataset that loads pre-recorded clean/noisy pairs."""
        # Resolve environment variables in paths
        from tf_restormer._config import expand_env_vars
        noisy_dir = expand_env_vars(dataset_config['noisy_dir'])
        clean_dir = expand_env_vars(dataset_config['clean_dir'])

        # noisy_suffix: e.g. "_ch1" for REVERB challenge (filters noisy files by suffix)
        self.noisy_suffix = dataset_config.get('noisy_suffix', '')
        if self.noisy_suffix:
            noisy_list = glob(os.path.join(noisy_dir, '**', f'*{self.noisy_suffix}.wav'), recursive=True)
            noisy_list += glob(os.path.join(noisy_dir, '**', f'*{self.noisy_suffix}.flac'), recursive=True)
        else:
            noisy_list = glob(os.path.join(noisy_dir, '**', '*.wav'), recursive=True)
            noisy_list += glob(os.path.join(noisy_dir, '**', '*.flac'), recursive=True)
        noisy_list.sort(key=lambda p: relpath(p, start=noisy_dir))
        self.list_noisy = noisy_list

        if clean_dir != None:
            clean_list = glob(os.path.join(clean_dir, '**', '*.wav'), recursive=True)
            clean_list += glob(os.path.join(clean_dir, '**', '*.flac'), recursive=True)
            clean_list.sort(key=lambda p: relpath(p, start=clean_dir))
            assert len(clean_list) == len(noisy_list), "Clean and noisy lists must have the same length."
            self.list_clean = clean_list
        else:
            self.list_clean = None

        self.n_item = len(noisy_list)
        self.fs_src = dataset_config['sample_rate_src']
        self.fs_in = dataset_config['sample_rate_in']
        
        logger.info(f"Test dataset: {self.n_item} items loaded from {clean_dir} and {noisy_dir}")
        logger.info(f"Test dataset fs_src: {self.fs_src}, fs_in: {self.fs_in}")
        logger.info(f"Metrics are: {dataset_config['metrics']}")
        
    def __getitem__(self, idx: int) -> dict:

        if self.list_clean != None:
            clean_path = self.list_clean[idx]
            clean, fs_clean = audio_lib.load(clean_path, sr=None) #! for metric computation
            clean = torch.from_numpy(clean)
            clean = torch_resample(clean, orig_freq=fs_clean, new_freq=self.fs_src, rolloff=0.98, lowpass_filter_width=64)
            # clean = clean.numpy()
        else:
            clean = None

        noisy_path = self.list_noisy[idx]
        noisy_input, fs_noisy = audio_lib.load(noisy_path, sr=None) #! for model input
        noisy_input = torch.from_numpy(noisy_input)
        noisy_input = torch_resample(noisy_input, orig_freq=fs_noisy, new_freq=self.fs_in, rolloff=0.98, lowpass_filter_width=64)
        if self.fs_in > self.fs_src:
            raise ValueError(f"fs_in {self.fs_in} must be less than or equal to fs_src {self.fs_src}.")
        elif self.fs_in <= self.fs_src:
            noisy = torch_resample(noisy_input, orig_freq=self.fs_in, new_freq=self.fs_src, rolloff=0.98, lowpass_filter_width=64)
            if clean is not None:
                min_len = min(len(noisy), len(clean))
                clean = clean[:min_len]
                noisy = noisy[:min_len]

    
        if clean is None:
            return {"noisy_distort": noisy, 
                    "noisy_distort_input": noisy_input, 
                    'fs_in': self.fs_in, 
                    'fs_src': self.fs_src,
                    'file_name': os.path.splitext(os.path.basename(noisy_path))[0]}
        else:
            return {"clean": clean, 
                    "noisy_distort": noisy, 
                    "noisy_distort_input": noisy_input, 
                    'fs_in': self.fs_in, 
                    'fs_src': self.fs_src,
                    'file_name': os.path.splitext(os.path.basename(noisy_path))[0]}

    def __len__(self) -> int:
        return self.n_item