import torch
import numpy as np

from math import ceil
from itertools import permutations
from dataclasses import dataclass, field, fields
from loguru import logger
from utils.decorators import *
from utils.util_stft import STFT, iSTFT
from utils.util_engine import p_law_compress
from transformers import WhisperModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torchaudio.transforms import MelSpectrogram, Resample
from torch_pesq import PesqLoss


# Utility functions
def l2norm(mat, keepdim=False, reduce='mean', squared=False):
    out = torch.norm(mat, dim=-1, keepdim=keepdim)
    if squared: out = out**2
    return out if reduce == 'sum' else out / mat.shape[-1]
	

def l1norm(mat, keepdim=False, reduce='mean'):
    out = torch.norm(mat, dim=-1, keepdim=keepdim, p=1)
    return out if reduce == 'sum' else out / mat.shape[-1]



class SSL_FM_Loss(torch.nn.Module):
	def __init__(self, model_key, resampler, device):
		super().__init__()
  
		model = Wav2Vec2Model.from_pretrained(model_key).to(device)

		self.feat_extractor = model.feature_extractor
		self.feat_extractor.eval()
		if resampler['orig_freq'] != resampler['new_freq']:
			self.resampler = Resample(orig_freq=resampler['orig_freq'],
									 new_freq=resampler['new_freq']).to(device)

	def normalize(self, x):
		mean = x.mean(dim=1, keepdim=True)
		std = x.std(dim=1, keepdim=True).clamp(min=1e-5)
		x = (x - mean) / std
	
		return x

	def forward(self, out, src):
		out = self.normalize(out)
		src = self.normalize(src)
		if hasattr(self, 'resampler'):
			out = self.resampler(out)
			src = self.resampler(src)
		with torch.no_grad():
			src_em = self.feat_extractor(src)
		out_em = self.feat_extractor(out)

		mse_loss = torch.norm(out_em - src_em, dim=(-1,-2))**2 / (out_em.shape[-1] * out_em.shape[-2])
		# mse_loss = l2norm(l2norm(out_em-src_em), squared=True)
		return torch.mean(mse_loss)


@logger_wraps()
class MS_STFT_L1_complex(torch.nn.Module):
	def __init__(self, alpha, beta, window_size=[256, 512, 768, 1024, 2048], is_log=True, device='cuda'):
		super().__init__()
		self.device = device
		self.alpha = alpha
		self.beta = beta
		stft_configs = [(win, win//4) for win in window_size]

		self.stft = [STFT(win, shift, device=device, normalize=True) for win, shift in stft_configs] 
		self.is_log = is_log
		self.tau = 1.0e-3
	def forward(self, out_wav, src_wav):
		loss_multi_res, loss_multi_res_r, loss_multi_res_i = [], [], []
		for stft in self.stft:
			out = stft(out_wav, cplx=True)
			src = stft(src_wav, cplx=True)
			if self.is_log:
				dist_abs = torch.log(abs(torch.abs(src) - torch.abs(out))/self.tau+1)
				dist_real = torch.log(abs(src.real - out.real)/self.tau+1)
				dist_imag = torch.log(abs(src.imag - out.imag)/self.tau+1)
			else:
				dist_abs = abs(torch.abs(src) - torch.abs(out))
				dist_real = abs(src.real - out.real)
				dist_imag = abs(src.imag - out.imag)
			loss_multi_res.append(torch.mean(dist_abs))
			loss_multi_res_r.append(torch.mean(dist_real))
			loss_multi_res_i.append(torch.mean(dist_imag))
		loss_multi_res = sum(loss_multi_res) / len(loss_multi_res)
		loss_multi_res_r = sum(loss_multi_res_r) / len(loss_multi_res_r)
		loss_multi_res_i = sum(loss_multi_res_i) / len(loss_multi_res_i)

		M_loss = loss_multi_res
		RI_loss = (loss_multi_res_r + loss_multi_res_i)/2
		loss =  (1-self.alpha)*RI_loss + self.alpha*M_loss
		return self.tau*loss if self.is_log else loss


@logger_wraps()
class MS_STFT_Gen_SC_Loss(torch.nn.Module):
	def __init__(self, window_size=[256, 512, 768, 1024, 2048], tau=1.0e-4, alpha=None, device='cuda'):
		super().__init__()
		self.device = device
		stft_configs = [(win, win//4) for win in window_size]

		self.stft = [STFT(win, shift, device=device, normalize=True) for win, shift in stft_configs] 
		self.tau = tau
		self.alpha = alpha

	def forward(self, out_wav, src_wav, epoch=1):
		if (self.alpha != None) and (epoch > 1):
			tau = max(self.alpha**(epoch-1), self.tau)
		else:
			tau = self.tau

		def gsc_loss(dist, src_abs, tau):
			# Generative Spectral Convergence Loss
			src_abs = torch.clamp(src_abs, min=tau)
			return torch.mean(src_abs * torch.log1p(dist / src_abs))

		loss_multi_res, loss_multi_res_r, loss_multi_res_i = [], [], []
		for i, stft in enumerate(self.stft):
			out = stft(out_wav, cplx=True)
			src = stft(src_wav, cplx=True)
			dist_abs = abs(torch.abs(src) - torch.abs(out))
			dist_real = abs(src.real - out.real)
			dist_imag = abs(src.imag - out.imag)
		
			# src_abs = (torch.abs(src)/torch.sqrt(torch.mean(torch.abs(src)**2,dim=-1,keepdim=True) + 1.0e-8))
			src_abs = torch.sqrt(torch.mean(torch.abs(src)**2,dim=-1,keepdim=True)) # B, 1, 1

			loss_multi_res.append(gsc_loss(dist_abs, src_abs, tau))
			loss_multi_res_r.append(gsc_loss(dist_real, src_abs, tau))
			loss_multi_res_i.append(gsc_loss(dist_imag, src_abs, tau))

		loss_multi_res = sum(loss_multi_res) / len(loss_multi_res)
		loss_multi_res_r = sum(loss_multi_res_r) / len(loss_multi_res_r)
		loss_multi_res_i = sum(loss_multi_res_i) / len(loss_multi_res_i)

		M_loss = loss_multi_res
		RI_loss = (loss_multi_res_r + loss_multi_res_i)/2
		loss =  0.4*RI_loss + 0.6*M_loss
		return loss


@logger_wraps()
class Time_Domain_L1(torch.nn.Module):
	def __init__(self, beta, device='cuda'):
		super().__init__()
		self.device = device
		self.beta = beta
  
	def forward(self, out_wav, src_wav):

		dist = abs(out_wav - src_wav)
		loss = torch.mean(self.beta*torch.log1p(dist/self.beta))

		return loss


class PESQ_Loss(torch.nn.Module):
	def __init__(self, fs=16000, device='cuda'):
		super().__init__()
		self.loss_pesq = PesqLoss(factor=1.0, sample_rate=fs).to(device)


	def forward(self, src_wav, out_wav):
		pesq = self.loss_pesq(src_wav, out_wav)
		return torch.mean(pesq)


class UTMOS_Loss(torch.nn.Module):
	def __init__(self, fs, device='cuda'):
		super().__init__()
		# Load UTMOS22 strong model via torch.hub
		self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
		self.model.to(device)
		self.model.train()
		# self.model.eval()
		self.fs = fs
	
	def forward(self, x):
		# Ensure inputs are in the correct shape (batch_size, samples)
		if x.dim() == 1:
			x = x.unsqueeze(0)
		
		scores = self.model(x, sr=self.fs)
		
		return -torch.mean(scores)
	
	def mos(self, x):
		# Ensure inputs are in the correct shape (batch_size, samples)
		if x.dim() == 1:
			x = x.unsqueeze(0)
		
		with torch.no_grad():
			scores = self.model(x, sr=self.fs)
		
		return scores


class HF_Loss(torch.nn.Module):
	def __init__(self, fs, device='cuda'):
		super().__init__()
		self.pesq_loss = PESQ_Loss(fs, device=device)
		self.utmos_loss = UTMOS_Loss(fs, device=device)
		
	def forward(self, src_wav, out_wav):
		pesq_loss = self.pesq_loss(src_wav, out_wav)
		utmos_loss = self.utmos_loss(out_wav)
		
		return pesq_loss + 10*utmos_loss