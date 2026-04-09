import os
import logging
import inspect
import numpy as np
from dotenv import load_dotenv
from scipy.signal import find_peaks

# Load environment variables
load_dotenv()

# Call logger for monitoring
logger = logging.getLogger("sys_log")

def parse_scps(scp_path, db_root=None):
    """Parse SCP file and optionally join paths with DB_ROOT.
    
    Args:
        scp_path: Path to SCP file
        db_root: Optional database root path. If provided, relative paths in SCP 
                 will be joined with this root.
    """
    scp_dict = dict()
    try:
        if not os.path.exists(scp_path): raise FileNotFoundError(f"File not found: {scp_path}")
        with open(scp_path, 'r') as f:
            for scp in f:
                scp_tokens = scp.strip().split()
                if len(scp_tokens) != 2: raise RuntimeError(f"Error format of context '{scp}'")
                key, addr = scp_tokens
                if key in scp_dict: raise ValueError(f"Duplicate key '{key}' exists!")
                
                # If db_root is provided and path is relative, join them
                if db_root and not os.path.isabs(addr):
                    addr = os.path.join(db_root, addr)
                
                scp_dict[key] = addr
    except (RuntimeError, ValueError) as e:
        logger.error(e)
        raise
    finally:
        func_name = inspect.currentframe().f_code.co_name
        logger.debug(f"Complete {__name__}.{func_name}")
    return scp_dict


def find_peak(rir, fs):
    alpha = 0.9  # penalty
    search_limit = int(2000*(fs/16000))
    search_rir = rir[:search_limit]
    peak_val = np.max(search_rir)
    peaks, _ = find_peaks(search_rir, height=peak_val*0.5)
    scores = search_rir[peaks] * np.exp(-alpha * peaks)
    peak_idx = peaks[np.argmax(scores)]
    return peak_idx

def rir_load(rir_dir, RT_list):
    h = {}
    for room in ['medium', 'small']:
        h[room] = []
        for idx, radius in enumerate([0.04, 0.0425, 0.045]):
        # for idx, radius in enumerate([0.04, 0.0425, 0.045, 0.0475, 0.05]):
            h[room].append([])
            for RT in RT_list:
                h[room][idx].append(np.load(rir_dir+'/'+room+'/mic_r_'+str(radius)+'/target_RT_'+str(RT)+'.npy'))
    return [h, RT_list]


def match_length(wav, max_len, idx_start=None) : 
    if len(wav) > max_len : 
        left = len(wav) - max_len
        if idx_start is None :
            idx_start = np.random.randint(left)
        wav = wav[idx_start:idx_start+max_len]
    elif len(wav) < max_len : 
        shortage = max_len - len(wav)
        pad_left = np.random.randint(0, shortage + 1)   # 0‥shortage
        pad_right = shortage - pad_left
        wav = np.pad(wav, (pad_left, pad_right))
    
    return wav, idx_start

def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar: scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar



import librosa
import scipy

maxv = np.iinfo(np.int16).max

def make_pcs(fs: int, nfft: int, PCS_min: float = 1.0000) -> np.ndarray:
 
    nyq = fs / 2
    # Select band edges and gain offsets based on sampling rate
    if nyq == 4000:
        edges = [0, 100, 200, 300, 400, 4000]
        gains_offset = [0.0000, 0.0702, 0.1825, 0.2877, 0.4000]
    elif nyq == 8000:
        edges = [0, 100, 200, 300, 400, 4400, 5300, 6400, 7700, 8000]
        gains_offset = [0.0000, 0.0702, 0.1825, 0.2877, 0.4000, 0.3228, 0.2386, 0.1614, 0.0772]
    else:
        edges = [0, 100, 200, 300, 400, 4400, 5300, 6400, 7700, 9500, nyq]
        gains_offset = [0.0000, 0.0702, 0.1825, 0.2877, 0.4000, 0.3228, 0.2386, 0.1614, 0.0772, 0.0772]
        
    # frequency axis for FFT bins
    freqs = np.linspace(0, fs//2, nfft//2 + 1)
    pcs = np.ones_like(freqs)

    # assign gains per band
    for g, lo, hi in zip(gains_offset, edges[:-1], edges[1:]):
        pcs[(freqs >= lo) & (freqs < hi)] = g + PCS_min

    return pcs



def Sp_and_phase(signal, fs, n_fft, PCS_min=1.0000):
    
    signal_length = signal.shape[0]
    
    hop_length = n_fft // 4
    PCS_gamma = make_pcs(fs, n_fft,  PCS_min=PCS_min)
    y_pad = librosa.util.fix_length(signal, size=signal_length + n_fft // 2)

    F = librosa.stft(y_pad, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=scipy.signal.windows.hamming)
    Lp = PCS_gamma * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    NLp = np.transpose(Lp, (1, 0))

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length, n_fft):
    mag = np.expm1(mag)
    Rec = np.multiply(mag, np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=n_fft//4,
                           win_length=n_fft,
                           window=scipy.signal.windows.hamming, length=signal_length)
    return result



def PCS(x, fs, PCS_min=1.000):
    assert fs in [8000, 16000, 48000], "sample rate must be either 8k, 16k, or 48kHz"

    nfft = int(512*(fs/16000))
    mag, phase, x_len = Sp_and_phase(x, fs, nfft, PCS_min)
    y = SP_to_wav(mag, phase, x_len, nfft)
    return y