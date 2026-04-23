import argparse
import functools
import os
import pickle
import random
import shutil
import subprocess
import tempfile
from glob import glob
from os.path import relpath

import colorednoise
import librosa as audio_lib
import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch
from loguru import logger
from pedalboard import Clipping
from scipy.signal import filtfilt, firwin2
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import resample as torch_resample

from tf_restormer.utils import util_dataset
from tf_restormer.utils.decorators import logger_wraps


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


@functools.lru_cache(maxsize=1)
def _check_ffmpeg() -> bool:
    """Return True if ffmpeg binary is available on PATH (result is cached)."""
    return shutil.which("ffmpeg") is not None


@functools.lru_cache(maxsize=1)
def _check_ffmpeg_filters(filter_names: tuple[str, ...]) -> frozenset[str]:
    """Return the subset of filter_names that ffmpeg supports (result is cached).

    Args:
        filter_names: Tuple of ffmpeg filter names to check (must be hashable for lru_cache).

    Returns:
        frozenset of available filter names from the requested set.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters", "-v", "quiet"],
            capture_output=True, text=True, timeout=10,
        )
        available = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            # Filter list line format: " ... filtername ..."
            # Lines that describe filters begin with a flag character (e.g. "A", "V", ".")
            parts = line.split()
            if len(parts) >= 2 and parts[1] in filter_names:
                available.add(parts[1])
        return frozenset(available)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("ffmpeg filter list parsing failed: %s", exc)
        return frozenset()


@functools.lru_cache(maxsize=1)
def _check_ffmpeg_encoders(encoder_names: tuple[str, ...]) -> frozenset[str]:
    """Return the subset of encoder_names that ffmpeg supports (result is cached).

    Args:
        encoder_names: Tuple of ffmpeg encoder names to check (must be hashable for lru_cache).

    Returns:
        frozenset of available encoder names from the requested set.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders", "-v", "quiet"],
            capture_output=True, text=True, timeout=10,
        )
        available = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if len(line) > 7 and line[0] in "VASD" and line[1] == ".":
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "=":
                    available.append(parts[1])
        return frozenset(n for n in encoder_names if n in available)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("ffmpeg encoder list parsing failed: %s", exc)
        return frozenset()


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

        # ------ Audio effects (ffmpeg-based digital distortion) ------ #
        self.available_filters: frozenset[str] = frozenset()
        self.available_encoders: frozenset[str] = frozenset()
        self.prob_effect = synthesis_config.get('audio_effects', {})
        self.has_ffmpeg = _check_ffmpeg()
        if self.prob_effect and not self.has_ffmpeg:
            logger.warning("ffmpeg not found. Digital distortion effects will be skipped.")
        elif self.prob_effect:
            needed_filters = ('crystalizer', 'flanger', 'acrusher')
            self.available_filters = _check_ffmpeg_filters(needed_filters)
            needed_encoders = ('libmp3lame', 'libvorbis', 'libopus')
            self.available_encoders = _check_ffmpeg_encoders(needed_encoders)


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

    def _resample(self, src):
        srcs_down = audio_lib.resample(src, orig_sr=self.fs, target_sr=self.fs_in)
        return srcs_down

    def _apply_ffmpeg_effect(self, audio: np.ndarray, sr: int, effect_str: str) -> np.ndarray:
        """Apply an ffmpeg audio filter chain to a numpy float32 array.

        Writes audio to a temp WAV, runs ffmpeg -af {effect_str}, reads back the result.
        On any failure, logs a warning and returns the original audio unchanged.

        Args:
            audio: Input float32 numpy array (mono).
            sr: Sample rate in Hz.
            effect_str: ffmpeg -af filter string, e.g. 'crystalizer=i=2,flanger=depth=3'.

        Returns:
            Processed float32 numpy array, same length as input.
        """
        original_len = len(audio)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, "input.wav")
                out_path = os.path.join(tmpdir, "output.wav")
                sf.write(in_path, audio, sr, subtype="FLOAT")
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", in_path,
                    "-af", effect_str,
                    "-codec:a", "pcm_f32le",
                    out_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode != 0:
                    logger.warning(
                        "_apply_ffmpeg_effect: ffmpeg failed (effect=%s, returncode=%d). "
                        "Returning original audio.\nstderr: %s",
                        effect_str,
                        result.returncode,
                        result.stderr.decode(errors="replace")[:400],
                    )
                    return audio
                processed, _ = sf.read(out_path, dtype="float32")
                if processed.ndim == 2:
                    processed = processed[:, 0]
                if len(processed) >= original_len:
                    processed = processed[:original_len]
                else:
                    processed = np.pad(processed, (0, original_len - len(processed)), mode="constant")
                return processed.astype(np.float32)
        except subprocess.TimeoutExpired:
            logger.warning("_apply_ffmpeg_effect: ffmpeg timeout (effect=%s). Returning original.", effect_str)
            return audio
        except Exception as exc:  # noqa: BLE001
            logger.warning("_apply_ffmpeg_effect: exception (effect=%s): %s. Returning original.", effect_str, exc)
            return audio

    def _apply_ffmpeg_codec(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply lossy codec encode-decode cycle (MP3 or OGG) via ffmpeg subprocess.

        Random consumption sequence mirrors engine.py L172:
          1st: random.choice(['mp3', 'ogg'])
          2nd: random.randint(4, 16) * 1000  (if mp3)  or
               random.choice(['vorbis', 'opus'])  (if ogg)

        Both random calls are always made, even when the encoder is unavailable,
        to preserve the global random state for reproducibility.

        Args:
            audio: Input float32 numpy array (mono).
            sr: Sample rate in Hz.

        Returns:
            Codec-processed float32 numpy array, same length as input.
        """
        original_len = len(audio)
        codec_type = random.choice(['mp3', 'ogg'])  # 1st random consumption
        if codec_type == 'mp3':
            bit_rate = random.randint(4, 16) * 1000  # 2nd random consumption
            encoder = 'libmp3lame'
            enc_ext = 'mp3'
        else:
            ogg_encoder = random.choice(['vorbis', 'opus'])  # 2nd random consumption
            encoder = f'lib{ogg_encoder}'  # libvorbis or libopus
            enc_ext = 'ogg'

        # Skip actual encoding if encoder is unavailable
        if encoder not in self.available_encoders:
            return audio

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, "input.wav")
                enc_path = os.path.join(tmpdir, f"encoded.{enc_ext}")
                dec_path = os.path.join(tmpdir, "decoded.wav")
                sf.write(in_path, audio, sr, subtype="FLOAT")

                if codec_type == 'mp3':
                    enc_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
                        "-codec:a", "libmp3lame",
                        "-b:a", str(bit_rate),
                        enc_path,
                    ]
                elif encoder == 'libvorbis':
                    enc_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
                        "-codec:a", "libvorbis",
                        enc_path,
                    ]
                else:  # opus
                    enc_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
                        "-codec:a", "libopus",
                        enc_path,
                    ]

                dec_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error", "-i", enc_path,
                    "-ar", str(sr),
                    "-codec:a", "pcm_f32le",
                    dec_path,
                ]

                enc_result = subprocess.run(enc_cmd, capture_output=True, timeout=30)
                if enc_result.returncode != 0:
                    logger.warning(
                        "_apply_ffmpeg_codec: encode failed (codec=%s/%s, returncode=%d). "
                        "Returning original.\nstderr: %s",
                        codec_type, encoder,
                        enc_result.returncode,
                        enc_result.stderr.decode(errors="replace")[:400],
                    )
                    return audio

                dec_result = subprocess.run(dec_cmd, capture_output=True, timeout=30)
                if dec_result.returncode != 0:
                    logger.warning(
                        "_apply_ffmpeg_codec: decode failed (codec=%s/%s, returncode=%d). "
                        "Returning original.\nstderr: %s",
                        codec_type, encoder,
                        dec_result.returncode,
                        dec_result.stderr.decode(errors="replace")[:400],
                    )
                    return audio

                decoded, _ = sf.read(dec_path, dtype="float32")
                if decoded.ndim == 2:
                    decoded = decoded[:, 0]
                if len(decoded) >= original_len:
                    decoded = decoded[:original_len]
                else:
                    decoded = np.pad(decoded, (0, original_len - len(decoded)), mode="constant")
                return decoded.astype(np.float32)

        except subprocess.TimeoutExpired:
            logger.warning("_apply_ffmpeg_codec: ffmpeg timeout (codec=%s/%s). Returning original.", codec_type, encoder)
            return audio
        except Exception as exc:  # noqa: BLE001
            logger.warning("_apply_ffmpeg_codec: exception (codec=%s/%s): %s. Returning original.", codec_type, encoder, exc)
            return audio

    def audio_effecter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply probabilistic digital distortion effects (crystalizer, flanger, acrusher, codec).

        Replicates the effect pipeline of engine.py audio_effecter() / random_effect()
        using ffmpeg subprocess instead of torchaudio.io.AudioEffector.

        Effect application order:
          1. crystalizer (if available) — applied via ffmpeg -af
          2. flanger    (if available) — applied via ffmpeg -af
          3. acrusher   (if available) — applied via ffmpeg -af
          4. codec      (mp3 or ogg)  — applied separately after filter chain

        IMPORTANT: random.random() calls for effects 1-3 are always made regardless of
        filter availability, to preserve the global random state.

        Args:
            audio: Input float32 numpy array (mono, 16kHz).
            sr: Sample rate in Hz (expected 16000).

        Returns:
            Distortion-applied float32 numpy array, same length as input.
        """
        prob = self.prob_effect
        audio = audio.astype(np.float32)

        # Build filter chain for effects 1-3
        # IMPORTANT: random parameter calls must ALWAYS execute (even if filter unavailable)
        # to preserve the global random state sequence.
        af_parts = []
        if random.random() < prob.get('crystalizer', 0.0):
            intensity = random.uniform(1, 4)  # always consume random
            if 'crystalizer' in self.available_filters:
                af_parts.append(f'crystalizer=i={intensity}')
        if random.random() < prob.get('flanger', 0.0):
            depth = random.uniform(1, 5)  # always consume random
            if 'flanger' in self.available_filters:
                af_parts.append(f'flanger=depth={depth}')
        if random.random() < prob.get('crusher', 0.0):
            bits = random.randint(1, 9)  # always consume random
            if 'acrusher' in self.available_filters:
                af_parts.append(f'acrusher=bits={bits}')

        # Apply all filter-chain effects in a single ffmpeg call
        if af_parts:
            audio = self._apply_ffmpeg_effect(audio, sr, ','.join(af_parts))

        # Apply codec separately (always last)
        if random.random() < prob.get('codec', 0.0):
            audio = self._apply_ffmpeg_codec(audio, sr)

        return audio


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

        # Digital distortion effects (crystalizer, flanger, acrusher, codec) via ffmpeg
        if self.prob_effect and self.has_ffmpeg:
            noisy_distort = self.audio_effecter(noisy_distort, self.fs)

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