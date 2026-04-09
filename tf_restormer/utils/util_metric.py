import pesq
import torch
import numpy as np
from .util_mcd import calculate as calculate_mcd
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources

"""
    output : wav[n_target,n_sample]
    target : wav[n_target,n_sample]
"""
def SIR(estim,target, requires_grad=False,device="cuda:0") :
    if estim.shape != target.shape : 
        raise Exception("ERROR::metric.py::SIR:: output shape != target shape | {} != {}".format(output.shape,target.shape))

    if len(estim.shape) != 2 : 
        raise Exception("ERROR::metric.py::SIR:: output dim {} != 2".format(len(output.shape)))
    n_target  = estim.shape[0]
    
    s_target = []
    e_interf = []

    for i in range(n_target) : 
        s_target.append(torch.inner(estim[i],target[i])*target[i]/torch.inner(target[i],target[i]))

        tmp = None
        for j in range(n_target) : 
            if i == j :
                continue
            if tmp is None : 
                tmp = torch.inner(estim[i],target[j])*target[j]/torch.inner(target[j],target[j])
            else : 
                tmp += torch.inner(estim[i],target[j])*target[j]/torch.inner(target[j],target[j])
        e_interf.append(tmp)

    SIR =  torch.tensor(0.0, requires_grad=requires_grad).to(device)
    for i in range(n_target) : 
        SIR += (torch.inner(s_target[i],s_target[i]))/torch.inner(e_interf[i],e_interf[i])
    return 10*torch.log10(SIR)
"""

"""
def PESQ(estim,target,fs=16000,mode="wb") :
    if torch.is_tensor(estim) : 
        estim = estim.cpu().detach().numpy()
    if torch.is_tensor(target) : 
        target = target.cpu().detach().numpy()

    if mode =="wb" : 
        val_pesq = pesq.pesq_batch(fs, target, estim, 'wb',on_error=pesq.PesqError.RETURN_VALUES,n_processor=20)
        val_pesq = sum(val_pesq)/len(val_pesq)
    elif mode == "nb" :
        val_pesq = pesq.pesq_batch(fs, target, estim, 'nb',on_error=pesq.PesqError.RETURN_VALUES,n_processor=20)
        val_pesq = sum(val_pesq)/len(val_pesq)
    else :
        val_pesq = pesq.pesq_batch(fs, target, estim, 'wb',on_error=pesq.PesqError.RETURN_VALUES,n_processor=20)
        val_pesq += pesq.pesq_batch(fs,target,estim,'nb',on_error=pesq.PesqError.RETURN_VALUES,n_processor=20)
        val_pesq = sum(val_pesq)/len(val_pesq)
    return val_pesq

def STOI(estim,target,fs=16000,mode="wb") :
    estim = estim.squeeze()
    target = target.squeeze()
    if torch.is_tensor(estim) : 
        estim = estim.cpu().detach().numpy()
    if torch.is_tensor(target) : 
        target = target.cpu().detach().numpy()

    return stoi(target, estim, fs, extended=False)


def MCD(ref, inf, fs, eps=1.0e-08):
    """Calculate Mel Cepstral Distortion (MCD).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
        eps (float): epsilon value for numerical stability
    Returns:
        mcd (float): MCD value between [0, +inf)
    """
    inf = inf.squeeze()
    ref = ref.squeeze()
    if torch.is_tensor(inf) : 
        inf = inf.cpu().detach().numpy()
    if torch.is_tensor(ref) : 
        ref = ref.cpu().detach().numpy()
    scaling_factor = np.sum(ref * inf) / (np.sum(inf**2) + eps)
    return calculate_mcd(ref, inf * scaling_factor, fs)

def LSD(estim, target, fs=16000):

    def stft(audio, n_fft=2048, hop_length=512):
        hann_window = torch.hann_window(n_fft).to(audio.device)
        stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
        stft_mag = torch.abs(stft_spec)
        stft_pha = torch.angle(stft_spec)

        return stft_mag, stft_pha

    sp = torch.log10(stft(estim)[0].square().clamp(1e-8))
    st = torch.log10(stft(target)[0].square().clamp(1e-8))
    lsd = (sp - st).square().mean(dim=1).sqrt().mean()

    return lsd.cpu().detach().numpy()

def SNR(estim,target, fs=16000, is_sdr=True) :
    if estim.shape != target.shape : 
        raise Exception("ERROR::metric.py::SIR:: output shape != target shape | {} != {}".format(estim.shape,target.shape))
    
    if is_sdr:
        if torch.is_tensor(estim) : 
            estim = estim.cpu().detach().numpy()
        if torch.is_tensor(target) : 
            target = target.cpu().detach().numpy()
        SDR, _, _, _ = bss_eval_sources(target, estim)
        return SDR
    else:
        estim = torch.Tensor(estim)
        target = torch.Tensor(target)
        s_target = (torch.inner(estim,target)*target/torch.inner(target,target))

        tmp = estim - s_target 
        e_noise = (tmp)
        SNR = (torch.inner(s_target,s_target))/torch.inner(e_noise,e_noise)
        return 10*torch.log10(SNR)



def run_metric(estim, target, key='PESQ', fs=16000):
    metric = globals()[key](estim,target,fs)
    
    return metric