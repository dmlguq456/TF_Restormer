import os
import torch
import csv
import time
from datetime import datetime
import soundfile as sf
import random
from loguru import logger
from tqdm import tqdm
import copy
import librosa
import shutil
from torch_pesq import PesqLoss
from utils import util_engine, util_stft, util_metric, util_writer, util_wvmos, util_dnsmos, util_sBERTscore
from utils.decorators import *
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from torchaudio.functional import resample as torch_resample
from torchaudio.io import AudioEffector, CodecConfig
from .loss import *
from .modules.msstftd import SFIMultiScaleSTFTDiscriminator 

class Engine(object):
    def __init__(self, args, config, model, dataloaders, gpuid, device):
        
        ''' Default setting '''
        self.args = args  # Store args for later use
        self.engine_mode = args.engine_mode
        self.config = config
        self.loader_config = config["dataloader"]
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        self.subset_conf = {}
        self.subset_conf["train"] = config["engine"]["subset"]["train"]
        self.subset_conf["valid"] = config["engine"]["subset"]["valid"]
        self.fs_src = config["dataset"][config["dataset_phase"]]["sample_rate_src"]
        self.fs_in = config["dataset"][config["dataset_phase"]]["sample_rate_in"]

        # pretrain_to16k / pretrain_to48k / adversarial_to16k / adversarial_to48k
        self.train_phase = config["train_phase"] + '_' + config["dataset_phase"]
        # loss configuration

        # STFT configuration
        self.fs_list = config['fs_list']
        self.fs_list_src = config['engine'][self.train_phase]['downsample_src']['fs_list_src']
        self.prob_downsample_src = config['engine'][self.train_phase]['downsample_src']['prob']
        self.stft, self.istft = {}, {}
        for fs in self.fs_list:
            frame_len = int(config['stft']['frame_length'] * int(fs) / 1000)
            frame_hop = int(config['stft']['frame_shift']  * int(fs) / 1000)
            self.stft[fs] = util_stft.STFT(frame_len, frame_hop, device=self.device, normalize=True)
            self.istft[fs] = util_stft.iSTFT(frame_len, frame_hop, device=self.device, normalize=True)
        self.out_F = int(config['stft']['frame_length'] * int(self.fs_src) / 1000) // 2 + 1

        # loss & simulation
        self.spec_aug = util_engine.RandSpecAugment(**self.config["engine"][self.train_phase]["RandSpecMasking"])
        self.w = config["engine"][self.train_phase]["loss_weight"]
        self.loss_t = Time_Domain_L1(**config["engine"][self.train_phase]["loss_time"], device=self.device)
        self.loss = MS_STFT_Gen_SC_Loss(**config["engine"][self.train_phase]["loss_enhance"], device=self.device)
        self.loss_fm = SSL_FM_Loss(**config["engine"][self.train_phase]["loss_rep"], device=self.device)
        self.loss_hf = HF_Loss(fs=16000, device=self.device)
        self.resample = T.Resample(orig_freq=16000, new_freq=8000)
        self.prob = config["engine"]["prob_effect"]

        # optim, scheduler, STFT configuration
        optim_cls = getattr(torch.optim, self.config["engine"]["optimizer"]["name"])
        sched_cls = getattr(torch.optim.lr_scheduler, self.config["engine"]["scheduler"]["name"])
        self.sample_file_list = config['engine']['sample_validation']


        # load enhance model
        config_name = self.args.config if hasattr(self.args, 'config') else 'default'
        log_base = f"log/log_{self.train_phase}_{config_name}"
        self.main_optimizer = optim_cls(self.model.parameters(),
                                        **self.config["engine"]["optimizer"].get(self.config["engine"]["optimizer"]["name"], {}))
        self.warmup_scheduler = util_engine.WarmupConstantSchedule(self.main_optimizer,
                                                                **self.config["engine"]["scheduler"]["WarmupConstantSchedule"])
        self.main_scheduler = sched_cls(self.main_optimizer,
                                        **self.config["engine"]["scheduler"].get(self.config["engine"]["scheduler"]["name"], {}))

        self.chkp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_base, "weights")
        os.makedirs(self.chkp_path, exist_ok=True)
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.chkp_path, self.model, self.main_optimizer, location=self.device)
        train_phase_list = config['train_phase_list']
        if 'adversarial' in self.train_phase:
            # Fallback from pretrain if no fine-tune checkpoints exist
            files = sorted(os.listdir(self.chkp_path))
            if not files:
                # locate pretrain checkpoint directory
                prev_stage = train_phase_list[train_phase_list.index(self.train_phase) - 1]

                prev_log_base = f"log_{prev_stage}_{config_name}"
                pretrain_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", prev_log_base, "weights")
                pre_files = sorted([f for f in os.listdir(pretrain_dir) if f.endswith('.pth')])
                if pre_files:
                    last_ckpt = pre_files[-1]
                    _ = util_engine.load_last_checkpoint_n_get_epoch(pretrain_dir, self.model, self.main_optimizer, location=self.device)
                    util_engine.save_checkpoint_per_nth(1, 0, self.model, self.main_optimizer, 1.0e5,  1.0e5, self.chkp_path)
                    logger.info(f"Copied pretrain checkpoint {last_ckpt} to fine-tune as epoch.0000.pth")
                # load or fallback checkpoint
                self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.chkp_path, self.model, self.main_optimizer, location=self.device)

        if 'adversarial' in self.train_phase:
            # discriminator
            self.msstftd = SFIMultiScaleSTFTDiscriminator(**config["engine"][self.train_phase]["msstftd"]).to(self.device)
            self.label_real = 1.0
            self.label_fake = 0.0

            # load discriminator model
            self.optimizer_disc = optim_cls(self.msstftd.parameters(),
                                            **self.config["engine"]["optimizer_D"].get(self.config["engine"]["optimizer_D"]["name"], {}))
            self.scheduler_disc = sched_cls(self.optimizer_disc, 
                                            **self.config["engine"]["scheduler"].get(self.config["engine"]["scheduler"]["name"], {}))
            self.chkp_path_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_base, "weights_D")
            os.makedirs(self.chkp_path_D, exist_ok=True)
            self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.chkp_path_D, self.msstftd, self.optimizer_disc, location=self.device)

        
        self.audio_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_base, "tensorboard")
        os.makedirs(self.audio_log_path, exist_ok=True)
        self.writer_src = util_writer.MyWriter(logdir=self.audio_log_path, 
                                               n_fft = int(config['stft']['frame_length'] * self.fs_src / 1000),
                                               n_hop = int(config['stft']['frame_shift'] * self.fs_src / 1000),
                                               sr=self.fs_src)

        # Logging 
        input_shape = tuple(self.stft[str(self.fs_in)](torch.randn(1, self.fs_in).to(self.device), cplx=True).squeeze(0).shape) + (2,) # (B, F, T, 2)
        util_engine.model_params_mac_summary(
            model=self.model, 
            input_shape=input_shape,
            metrics=['ptflops', 'thop', 'torchinfo'],
            device=self.device
        )


    def audio_effecter(self, audio, sample_rate, batch_wise=True):
        # codec effect

        assert sample_rate == 16000, "Currently only 16kHz is supported for audio_effecter"

        def mp3_case():
            bit_rate = int(random.randint(4,16)*1000)
            return AudioEffector(format='mp3', codec_config=CodecConfig(bit_rate=bit_rate))

        def ogg_case():
            encoder = random.choice(['vorbis', 'opus'])
            return AudioEffector(format='ogg', encoder=encoder)
        

        def apply_pipeline(x, sr):
            if not effects:
                return x

            for eff in effects:
                x = eff.apply(x, sr)
            return x

        def random_effect():
            effects = []
            if random.random() < self.prob['crystalizer']:
                intensity=random.uniform(1, 4)
                effects.append(AudioEffector(effect=f'crystalizer=i={intensity}'))
            if random.random() < self.prob['flanger']:
                depth = random.uniform(1, 5)
                effects.append(AudioEffector(effect=f'flanger=depth={depth}'))
            if random.random() < self.prob['crusher']:
                bits = random.randint(1, 9)
                effects.append(AudioEffector(effect=f'acrusher=bits={bits}'))
            # codec is applied at the end
            if random.random() < self.prob['codec']:
                effects.append(random.choice([mp3_case, ogg_case])())
            return effects
        
        # 배치 순회
        if batch_wise:
            processed = []
            for i in range(audio.shape[0]):
                effects = random_effect()
                sample = audio[i]
                effected = apply_pipeline(sample.unsqueeze(-1), sample_rate)
                effected = effected[:sample.shape[0], :]  # Ensure the output length matches the input
                processed.append(effected.squeeze(-1))
            audio = torch.stack(processed, dim=0)
        else:
            effects = random_effect()
            audio = apply_pipeline(audio.permute(1,0), sample_rate)
            audio = audio.permute(1,0)
        
        if random.random() < self.prob['downsample_8k']:
            sample_rate = 8000
            audio = torch_resample(audio, orig_freq=16000, new_freq=8000, lowpass_filter_width=32, rolloff=0.98)
        else:
            sample_rate = 16000
    
        
        return audio, sample_rate

    def target_downsample(self, target):
        if random.random() < self.prob_downsample_src:
            fs_target = int(random.choice(self.fs_list_src))
            target = torch_resample(target, orig_freq=self.fs_src, new_freq=fs_target, lowpass_filter_width=32, rolloff=0.98)
            out_F = int(self.config['stft']['frame_length'] * fs_target / 1000) // 2 + 1
        else:
            fs_target = self.fs_src
            out_F = self.out_F

        return target, fs_target, out_F

    def _discriminator_step(self, out, src, fs, update: bool):
        total = 0.0
        num_iter = 2 if update else 1
        for _ in range(num_iter):
            # fake
            logit_fake, _ = self.msstftd(out.detach(), fs)
            label_fake = [torch.full_like(lf, self.label_fake) for lf in logit_fake]
            loss_fake  = [torch.nn.functional.mse_loss(logit, label) for logit, label in zip(logit_fake, label_fake)]

            logit_real, _ = self.msstftd(src, fs)
            label_real = [torch.full_like(logit, self.label_real) for logit in logit_real]
            loss_real  = [torch.nn.functional.mse_loss(logit, label) for logit, label in zip(logit_real, label_real)]

            cur_loss_disc = torch.stack(loss_real).mean() + torch.stack(loss_fake).mean()
            if update:
                self.optimizer_disc.zero_grad()
                cur_loss_disc.backward()
                if self.config['engine']['clip_norm']: 
                    torch.nn.utils.clip_grad_norm_(self.msstftd.parameters(), self.config['engine']['clip_norm'])
                self.optimizer_disc.step()
        return cur_loss_disc


    def _generator_step(self, out, src, fs):
        # disable D-grad
        for p in self.msstftd.parameters(): p.requires_grad_(False)

        logit_fake_G, feat_fake = self.msstftd(out, fs)
        with torch.no_grad():
            _, feat_real = self.msstftd(src, fs)
        label_real_G = [torch.full_like(logit, self.label_real) for logit in logit_fake_G]
        # adversarial LS-GAN loss
        cur_loss_G_adv = [torch.nn.functional.mse_loss(lf, label) for lf, label in zip(logit_fake_G, label_real_G)]
        cur_loss_G_adv = torch.stack(cur_loss_G_adv).mean()
        # adversarial feature matching loss loss
        cur_loss_G_fm = [sum(torch.nn.functional.l1_loss(f, r) for f, r in zip(fake, real))
                                    for fake, real in zip(feat_fake, feat_real)]
        cur_loss_G_fm = torch.stack(cur_loss_G_fm).mean()

        # restore D-grad
        for p in self.msstftd.parameters(): p.requires_grad_(True)

        return cur_loss_G_adv, cur_loss_G_fm

    @logger_wraps()
    def _train(self, _dataloader, epoch):
        if self.subset_conf["train"]["subset"]:
            sampler = util_engine.create_sampler(len(_dataloader.dataset), self.subset_conf["train"]["num_per_epoch"])
            dataloader = DataLoader(dataset = _dataloader.dataset, collate_fn = _dataloader.collate_fn, sampler = sampler, **self.loader_config)
        else:
            dataloader = _dataloader

        self.model.train()
        tot_loss_time, tot_loss_se, tot_loss_rep, tot_loss_pesq, tot_loss_G_fm, tot_loss_G_adv, tot_loss_disc, n_batch = 0, 0, 0, 0, 0, 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True)
        for batch in dataloader:
            input_sizes = batch["num_sample"]
            target = batch["clean"].type(torch.float32)
            noisy = batch["noisy_distort"].type(torch.float32)
            # Scheduler learning rate for warm-up (Iteration-based update for transformers)
            if epoch == 1: self.warmup_scheduler.step()
            # feature pre-processing

            target, fs_target, out_F = self.target_downsample(target)
            target_stft = self.stft[str(fs_target)](target.to(self.device), cplx=True) # B, F, T

            noisy, fs_noisy = self.audio_effecter(noisy, 16000)
            noisy_stft = self.stft[str(fs_noisy)](noisy.to(self.device), cplx=True)  # B, F, T

            # spec augment for masked modelling
            noisy_stft, mask = self.spec_aug(noisy_stft, is_idx=True)

            # generator
            model_input = torch.stack([torch.real(noisy_stft), torch.imag(noisy_stft)],dim=-1) # B, F, T, 2
            out = self.model(model_input, out_F=out_F) # B, F, T, 2
            out = torch.complex(out[...,0], out[...,1])  # [M, F, T]
            
            out_wav = self.istft[str(fs_target)](out, cplx=True, squeeze=False) # B, F, T -> B, L
            src_wav = self.istft[str(fs_target)](target_stft, cplx=True) # B, F, T -> B, L
            
            # regression loss
            cur_loss_se = self.loss(out_wav, src_wav, epoch=epoch)
            tot_loss_se += cur_loss_se.item()
            cur_loss_time = self.loss_t(out_wav, src_wav)
            tot_loss_time += cur_loss_time.item()
            cur_loss_rep = self.loss_fm(out_wav, src_wav)
            tot_loss_rep += cur_loss_rep.item()

            # adversarial training + PESQ loss
            if 'pretrain' in self.train_phase: # only when pretraining w/o adversarial training
                cur_loss_gen = cur_loss_se*self.w['se'] + cur_loss_time*self.w['time'] + cur_loss_rep*self.w['ssl']
            else:
                src_wav_16k = torch_resample(src_wav, orig_freq=fs_target, new_freq=16000, lowpass_filter_width=32, rolloff=0.98) if fs_target != 16000 else src_wav
                out_wav_16k = torch_resample(out_wav, orig_freq=fs_target, new_freq=16000, lowpass_filter_width=32, rolloff=0.98) if fs_target != 16000 else out_wav
                cur_loss_pesq = self.loss_hf(src_wav_16k, out_wav_16k)
                tot_loss_pesq += cur_loss_pesq.item()
                cur_loss_disc = self._discriminator_step(out_wav, src_wav, fs_target, update=True)
                tot_loss_disc += cur_loss_disc.item()
                cur_loss_G_adv, cur_loss_G_fm = self._generator_step(out_wav, src_wav, fs_target)
                tot_loss_G_adv += cur_loss_G_adv.item()
                tot_loss_G_fm += cur_loss_G_fm.item()
                cur_loss_gen = (cur_loss_se*self.w['se'] + cur_loss_time*self.w['time'] + cur_loss_rep*self.w['ssl']
                                + cur_loss_G_adv*self.w['gan'] + cur_loss_G_fm*self.w['fm'] 
                                + cur_loss_pesq*self.w['pesq'])
                
                

            self.main_optimizer.zero_grad()
                
            cur_loss_gen.backward()
            if self.config['engine']['clip_norm']: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            self.main_optimizer.step()

            # update the progress bar
            n_batch += 1
            pbar.update(1)
            dict_loss = {
                'L_time': tot_loss_time/n_batch,
                'L_se': tot_loss_se/n_batch,
                'L_rep': tot_loss_rep/n_batch,
                'L_pesq': tot_loss_pesq/n_batch,
                'L_G_adv': tot_loss_G_adv/n_batch,
                'L_G_fm': tot_loss_G_fm/n_batch,
                'L_D_adv': tot_loss_disc/n_batch
            }
            pbar.set_postfix({k: f"{v:.2e}" for k, v in dict_loss.items()})
        pbar.close()

        return dict_loss, n_batch
    
    @logger_wraps()
    def _validate(self, _dataloader, epoch):
        if self.subset_conf["valid"]["subset"]:
            num_smmpl = self.subset_conf["valid"]["num_per_epoch"] if epoch > 0 else 100 
            sampler = util_engine.create_sampler(len(_dataloader.dataset), num_smmpl)
            dataloader = DataLoader(dataset = _dataloader.dataset, collate_fn = _dataloader.collate_fn, sampler = sampler, **self.loader_config)
        else:
            dataloader = _dataloader        

        self.model.eval()
        tot_loss_time, tot_loss_se, tot_loss_rep, tot_loss_pesq, tot_loss_G_fm, tot_loss_G_adv, tot_loss_disc, n_batch = 0, 0, 0, 0, 0, 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="RED", dynamic_ncols=True)
        
        random_sample_idx = 10
        cnt = 0
        with torch.inference_mode():
            for batch in dataloader:
                input_sizes = batch["num_sample"]
                target = batch["clean"].type(torch.float32)
                noisy = batch["noisy_distort"].type(torch.float32)

                # feature pre-processing
                target, fs_target, out_F = self.target_downsample(target)
                target_stft = self.stft[str(fs_target)](target.to(self.device), cplx=True) # B, F, T
                

                noisy, fs_noisy = self.audio_effecter(noisy, 16000)
                noisy_stft = self.stft[str(fs_noisy)](noisy.to(self.device), cplx=True)  # B, F, T
                
                # spec augment for masked modelling
                noisy_stft, mask = self.spec_aug(noisy_stft, is_idx=True)

                # generator
                model_input = torch.stack([torch.real(noisy_stft), torch.imag(noisy_stft)],dim=-1) # B, F, T, 2
                out = self.model(model_input, out_F=out_F)
                out = torch.complex(out[...,0], out[...,1])  # [M, F, T]

                out_wav = self.istft[str(fs_target)](out, cplx=True, squeeze=False) # B, F, T -> B, L
                src_wav = self.istft[str(fs_target)](target_stft, cplx=True) # B, F, T -> B, L
                # regression loss
                cur_loss_se = self.loss(out_wav, src_wav, epoch=epoch)
                tot_loss_se += cur_loss_se.item()
                cur_loss_time = self.loss_t(out_wav, src_wav)
                tot_loss_time += cur_loss_time.item()
                cur_loss_rep = self.loss_fm(out_wav, src_wav)
                tot_loss_rep += cur_loss_rep.item()
                # adversarial training + PESQ loss
                if 'adversarial' in self.train_phase:
                    src_wav_16k = torch_resample(src_wav, orig_freq=fs_target, new_freq=16000, lowpass_filter_width=32, rolloff=0.98) if fs_target != 16000 else src_wav
                    out_wav_16k = torch_resample(out_wav, orig_freq=fs_target, new_freq=16000, lowpass_filter_width=32, rolloff=0.98) if fs_target != 16000 else out_wav
                    cur_loss_pesq = self.loss_hf(src_wav_16k, out_wav_16k)
                    tot_loss_pesq += cur_loss_pesq.item()
                    cur_loss_disc = self._discriminator_step(out_wav, src_wav, fs_target, update=False)
                    tot_loss_disc += cur_loss_disc.item()
                    cur_loss_G_adv, cur_loss_G_fm = self._generator_step(out_wav, src_wav, fs_target)
                    tot_loss_G_adv += cur_loss_G_adv.item()
                    tot_loss_G_fm += cur_loss_G_fm.item()
                    

                # save randomly chosen sample to writer
                if n_batch == random_sample_idx:
                    self.logging_sample_wav(out, noisy_stft, target, fs_noisy, fs_target, epoch)

                # update the progress bar
                n_batch += 1
                pbar.update(1)
                dict_loss = {
                    'L_time': tot_loss_time/n_batch,
                    'L_se': tot_loss_se/n_batch,
                    'L_rep': tot_loss_rep/n_batch,
                    'L_pesq': tot_loss_pesq/n_batch,
                    'L_G_adv': tot_loss_G_adv/n_batch,
                    'L_G_fm': tot_loss_G_fm/n_batch,
                    'L_D_adv': tot_loss_disc/n_batch
                }
                pbar.set_postfix({k: f"{v:.2e}" for k, v in dict_loss.items()})
            pbar.close()

            return dict_loss, n_batch


    def logging_sample_wav(self, out, noisy_stft, target, fs_noisy, fs_target, epoch):

        def load_real_sample_infer(sample_file, idx):
            sample_x, fs = librosa.load(sample_file, sr=None)
            if fs > 16000:
                sample_x = librosa.resample(sample_x, orig_sr=fs, target_sr=16000)
                fs = 16000
            sample_x = torch.tensor(sample_x)[None] # 1, N
            sample_stft = self.stft[str(fs)](sample_x.to(self.device), cplx=True) # B, F, T
            sample_input = torch.stack([torch.real(sample_stft), torch.imag(sample_stft)],dim=-1) # B, T, F, 2
            sample_output = self.model(sample_input.to(self.device), out_F=self.out_F)
            sample_output = torch.complex(sample_output[...,0], sample_output[...,1])  # [M, F, T]
            sample_estim = self.istft[str(self.fs_src)](sample_output, cplx=True, squeeze=True)

            if fs < self.fs_src:
                sample_x = librosa.resample(sample_x.cpu().data.numpy(), orig_sr=fs, target_sr=self.fs_src)
                sample_x = torch.tensor(sample_x) # 1, N
                
            self.writer_src.log_wav2spec(sample_x[0], "sample_noisy_distort_"+str(idx), epoch)
            self.writer_src.log_audio(sample_x[0], 'sample_noisy_distort_audio_'+str(idx), epoch)

            self.writer_src.log_wav2spec(sample_estim, "sample_enhance_out_"+str(idx), epoch)
            self.writer_src.log_audio(sample_estim, 'sample_estim_enhance_audio_'+str(idx), epoch)   
        
        noisy = self.istft[str(fs_noisy)](noisy_stft[0]+1.0e-8, cplx=True, squeeze=True) # B, F, T
        noisy = librosa.resample(noisy.cpu().data.numpy(), orig_sr=fs_noisy, target_sr=self.fs_src)
        noisy = torch.tensor(noisy) # 1, N
        self.writer_src.log_wav2spec(noisy, "noisy_distort", epoch)
        self.writer_src.log_audio(noisy, 'noisy_distort_audio', epoch)

        estim_src = self.istft[str(fs_target)](out, cplx=True)
        estim_src = torch_resample(estim_src, orig_freq=fs_target, new_freq=self.fs_src, lowpass_filter_width=32, rolloff=0.98) if fs_target != self.fs_src else estim_src
        self.writer_src.log_wav2spec(estim_src[0], "enhance_out", epoch)
        self.writer_src.log_audio(estim_src[0], 'estim_enhance_audio', epoch)

        target = torch_resample(target, orig_freq=fs_target, new_freq=self.fs_src, lowpass_filter_width=32, rolloff=0.98) if fs_target != self.fs_src else target
        self.writer_src.log_wav2spec(target[0], "clean", epoch)
        self.writer_src.log_audio(target[0], 'clean_audio', epoch)

        # check real-recorded sample
        for idx, sample_ in enumerate(self.sample_file_list):
            load_real_sample_infer(sample_, idx)

        
    
    @logger_wraps()
    def run(self):
        with torch.cuda.device(self.device):
            init_loss_dict, _ = self._validate(self.dataloaders['valid'], self.start_epoch-1)
            logger.info((f"[INIT] Loss(time/mini-batch)\n Epoch {self.start_epoch-1:2d}: "
                        f"L_time = {init_loss_dict['L_time']:.2e} | "
                        f"L_se = {init_loss_dict['L_se']:.2e} | "
                        f"L_rep = {init_loss_dict['L_rep']:.2e} | "
                        f"L_pesq = {init_loss_dict['L_pesq']:.2e} | "
                        f"L_G_adv = {init_loss_dict['L_G_adv']:.2e} | "
                        f"L_G_fm = {init_loss_dict['L_G_fm']:.2e} | "
                        f"L_D = {init_loss_dict['L_D_adv']:.2e}"))
            
            for epoch in range(self.start_epoch, self.config['engine']['max_epoch'][self.train_phase] + 1):

                # Training                    
                train_start_time = time.time()
                tr_loss_dict, tr_n_batch = self._train(self.dataloaders['train'], epoch)
                train_end_time   = time.time()

                # Validation #! validation is always based on end-to-end (not oracle enhance or restore)
                val_start_time   = time.time()
                val_loss_dict, val_n_batch = self._validate(self.dataloaders['valid'], epoch)
                val_end_time     = time.time()
                
                # Scheduling
                if epoch > self.config['engine']['start_scheduling'][self.train_phase]:
                    self.main_scheduler.step()
                    if 'adversarial' in self.train_phase:
                        self.scheduler_disc.step()


                # Logging to terminal
                logger.info((f"[TRAIN] Loss(time/mini-batch)\n Epoch {epoch:2d}:"
                            f"L_time = {tr_loss_dict['L_time']:.2e} | "
                            f"L_se = {tr_loss_dict['L_se']:.2e} | "
                            f"L_rep = {tr_loss_dict['L_rep']:.2e} | "
                            f"L_pesq = {tr_loss_dict['L_pesq']:.2e} | "
                            f"L_G_adv = {tr_loss_dict['L_G_adv']:.2e} | "
                            f"L_G_fm = {tr_loss_dict['L_G_fm']:.2e} | "
                            f"L_D = {tr_loss_dict['L_D_adv']:.2e} | "
                            f"Speed = ({train_end_time - train_start_time:.2f}s/{tr_n_batch:d})"))
                logger.info((f"[VALID] Loss(time/mini-batch)\n Epoch {epoch:2d}:"
                            f"L_time = {val_loss_dict['L_time']:.2e} | "
                            f"L_se = {val_loss_dict['L_se']:.2e} | "
                            f"L_rep = {val_loss_dict['L_rep']:.2e} | "
                            f"L_pesq = {val_loss_dict['L_pesq']:.2e} | "
                            f"L_G_adv = {val_loss_dict['L_G_adv']:.2e} | "
                            f"L_G_fm = {val_loss_dict['L_G_fm']:.2e} | "
                            f"L_D = {val_loss_dict['L_D_adv']:.2e} | "
                            f"Speed = ({val_end_time - val_start_time:.2f}s/{val_n_batch:d})"))


                if 'pretrain' in self.train_phase: # only when pretraining w/o adversarial training
                    val_loss_gen = (val_loss_dict['L_se']*self.w['se'] + val_loss_dict['L_rep']*self.w['ssl'])
                    tr_loss_gen = (tr_loss_dict['L_se']*self.w['se'] + tr_loss_dict['L_rep']*self.w['ssl']) 
                else:
                    tr_loss_gen = (tr_loss_dict['L_se']*self.w['se'] + tr_loss_dict['L_rep']*self.w['ssl'] + tr_loss_dict['L_pesq']*self.w['pesq']
                                    + tr_loss_dict['L_G_adv']*self.w['gan'] + tr_loss_dict['L_G_fm']*self.w['fm'])
                    val_loss_gen = (val_loss_dict['L_se']*self.w['se'] + val_loss_dict['L_rep']*self.w['ssl'] + val_loss_dict['L_pesq']*self.w['pesq']
                                    + val_loss_dict['L_G_adv']*self.w['gan'] + val_loss_dict['L_G_fm']*self.w['fm'])
                # save checkpoint
                # val_loss_best = util_engine.save_checkpoint_per_best(val_loss_best, val_loss, tr_loss, epoch, self.model, self.main_optimizer, self.chkp_path)
                util_engine.save_checkpoint_per_nth(1, epoch, self.model, self.main_optimizer, tr_loss_gen, val_loss_gen, self.chkp_path)
                if 'adversarial' in self.train_phase:
                    util_engine.save_checkpoint_per_nth(1, epoch, self.msstftd, self.optimizer_disc, tr_loss_dict['L_D_adv'], val_loss_dict['L_D_adv'], self.chkp_path_D)

                # Logging to tensorboard (Tensorboard)
                self.writer_src.add_scalars("Loss_se", {'Train': tr_loss_dict['L_se'], 'Valid': val_loss_dict['L_se']}, epoch), 
                self.writer_src.add_scalars("Loss_time", {'Train': tr_loss_dict['L_time'], 'Valid': val_loss_dict['L_time']}, epoch), 
                self.writer_src.add_scalars("Loss_rep", {'Train': tr_loss_dict['L_rep'], 'Valid': val_loss_dict['L_rep']}, epoch),
                if self.train_phase != 'pretrain':
                    self.writer_src.add_scalars("Loss_G_adv", {'Train': tr_loss_dict['L_G_adv'], 'Valid': val_loss_dict['L_G_adv']}, epoch),
                    self.writer_src.add_scalars("Loss_G_fm", {'Train': tr_loss_dict['L_G_fm'], 'Valid': val_loss_dict['L_G_fm']}, epoch)
                    self.writer_src.add_scalars("Loss_D", {'Train': tr_loss_dict['L_D_adv'], 'Valid': val_loss_dict['L_D_adv']}, epoch)
                    self.writer_src.add_scalars("Loss_pesq", {'Train': tr_loss_dict['L_pesq'], 'Valid': val_loss_dict['L_pesq']}, epoch)
                self.writer_src.add_scalar("LearningRate", self.main_optimizer.param_groups[0]['lr'], epoch)
                self.writer_src.flush()

            logger.info(f"Training for {self.config['engine']['max_epoch']} epoches done!")
            
            # Close the writer to properly save all data and release resources
            self.writer_src.close()
            logger.info("TensorBoard writer closed successfully")

        

class EngineEval(object):
    def __init__(self, args, config, model, dataloaders, gpuid, device):

        ''' Default setting '''
        self.args = args  # Store args for later use
        self.engine_mode = args.engine_mode
        self.config = config
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        testset_key = self.config['dataset_test']['testset_key']
        self.input_eval = config["dataset_test"]['input_eval']
        self.output_eval = config["dataset_test"]['output_eval']
        self.fs_src = config['dataset_test'][testset_key]['sample_rate_src']
        self.fs_in = config['dataset_test'][testset_key]['sample_rate_in']
        self.metrics_key = config['dataset_test'][testset_key]['metrics']
        self.resampler = T.Resample(orig_freq=self.fs_src, new_freq=16000, 
                                    resampling_method='sinc_interp_hann',
                                    lowpass_filter_width=32,
                                    rolloff=0.98).to(self.device)
        # loss configuration
        self.train_phase = config["train_phase"] + '_' + config["dataset_phase"]
        self.w = config["engine"][self.train_phase]["loss_weight"]
        self.loss_t = Time_Domain_L1(**config["engine"][self.train_phase]["loss_time"], device=self.device)
        self.loss = MS_STFT_Gen_SC_Loss(**config["engine"][self.train_phase]["loss_enhance"], device=self.device)
        self.loss_fm = SSL_FM_Loss(**config["engine"][self.train_phase]["loss_rep"], device=self.device)
        # MOS metrics
        self.utmos = UTMOS_Loss(16000, device=self.device)
        self.wvmos = util_wvmos.Wav2Vec2MOS(device=self.device)
        # speechBERTscore
        self.bleu_scorer = util_sBERTscore.SpeechMetricCalculator(metric_type='bleu', device=self.device)
        self.bert_scorer = util_sBERTscore.SpeechMetricCalculator(metric_type='bertscore', device=self.device)
        self.token_dist_scorer = util_sBERTscore.SpeechMetricCalculator(metric_type='tokendistance', device=self.device)

        # optim, scheduler, STFT configuration
        optim_cls = getattr(torch.optim, self.config["engine"]["optimizer"]["name"])
        self.stft, self.istft = {}, {}
        self.fs_list = config['fs_list']
        for fs in self.fs_list:
            frame_len = int(config['stft']['frame_length'] * int(fs) / 1000)
            frame_hop = int(config['stft']['frame_shift'] * int(fs) / 1000)
            self.stft[fs] = util_stft.STFT(frame_len, frame_hop, device=self.device, normalize=True)
            self.istft[fs] = util_stft.iSTFT(frame_len, frame_hop, device=self.device, normalize=True)
        self.out_F = int(config['stft']['frame_length'] * int(self.fs_src) / 1000) // 2 + 1

        self.sample_file_list = config['engine']['sample_validation']
        # load enhance model
        config_name = self.args.config if hasattr(self.args, 'config') else 'default'
        log_base = f"log/log_{self.train_phase}_{config_name}"
        self.main_optimizer = optim_cls(self.model.parameters(),
                                        **self.config["engine"]["optimizer"].get(self.config["engine"]["optimizer"]["name"], {}))
        self.chkp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_base, "weights")
        os.makedirs(self.chkp_path, exist_ok=True)
        assert os.path.exists(self.chkp_path), f"Checkpoint path {self.chkp_path} does not exist!"
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.chkp_path, self.model, self.main_optimizer, location=self.device)
        
        self.audio_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_base, "tensorboard_eval_"+testset_key)
        os.makedirs(self.audio_log_path, exist_ok=True)
        self.writer_src = util_writer.MyWriter(logdir=self.audio_log_path, 
                                               n_fft=config['stft']['frame_length'] * self.fs_src // 16000, 
                                               n_hop=config['stft']['frame_shift'] * self.fs_src // 16000,
                                               sr=self.fs_src)

        self.random_sample_idx = self.config['dataset_test'][testset_key]['random_sample_idx']

        # Logging 
        input_shape = tuple(self.stft[str(self.fs_in)](torch.randn(1, self.fs_in).to(self.device), cplx=True).squeeze(0).shape) + (2,) # (B, F, T, 2)
        util_engine.model_params_mac_summary(
            model=self.model, 
            input_shape=input_shape, 
            metrics=['ptflops', 'thop', 'torchinfo'],
            device=self.device
        )

    @staticmethod
    def normalize(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-5)
        x = (x - mean) / std
        return x

    @staticmethod
    def peak_norm(x, eps=1e-12):
        return x / (torch.max(torch.abs(x)) + eps)    
        # return x / (torch.max(torch.abs(x)) + eps)    


    def _update_metrics(self, metrics, key, value_in=None, value_out=None):
        if key in self.metrics_key:
            if value_in != None:
                metrics[f'{key}_i'] += value_in
                metrics[f'{key}_i_2'] += value_in**2
            if value_out != None:
                metrics[f'{key}'] += value_out
                metrics[f'{key}_2'] += value_out**2    
    
    @logger_wraps()
    def _evaluate(self, dataloader, epoch):
        self.model.eval()
        from collections import defaultdict
        
        # 딕셔너리를 사용하여 모든 메트릭을 한 번에 초기화
        metrics = defaultdict(float)
        n_batch = 0
        
        pbar = tqdm(total=len(dataloader), unit='utt', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="WHITE", dynamic_ncols=True)
        
        with torch.inference_mode():
            for batch in dataloader:
                # target = batch["clean"].type(torch.float32)
                noisy = batch["noisy_distort"].type(torch.float32)
                noisy_orig = batch["noisy_distort_input"].type(torch.float32)
                target = batch.get("clean", None)
                fs_in = batch["fs_in"].item()
                fs_src = batch["fs_src"].item()
                # feature pre-processing
                if target is not None:
                    target = target.type(torch.float32)
                    target_stft = self.stft[str(fs_src)](target.to(self.device), cplx=True)  # B, F, T

                noisy_stft = self.stft[str(fs_src)](noisy.to(self.device), cplx=True)  # B, F, T
                noisy_orig_stft = self.stft[str(fs_in)](noisy_orig.to(self.device), cplx=True)  # B, F, T
                # generator
                wav_len_list = []
                wav_len_list_16k = []
                if self.input_eval:
                    in_wav_orig = self.istft[str(fs_in)](noisy_orig_stft, cplx=True, squeeze=False) # B, F, T -> B, L
                    in_wav = noisy.to(self.device)
                    # in_wav = self.istft[str(fs_src)](noisy_stft, cplx=True, squeeze=False)
                    in_wav_16k = self.resampler(in_wav)
                    in_wav_norm = self.normalize(in_wav_16k)
                    wav_len_list.append(in_wav.shape[-1])
                    wav_len_list_16k.append(in_wav_16k.shape[-1])

                if self.output_eval:
                    model_input = torch.stack([torch.real(noisy_orig_stft), torch.imag(noisy_orig_stft)],dim=-1) # B, F, T, 2
                    comp = self.model(model_input, out_F=self.out_F) # B, F, T, 2
                    out = torch.complex(comp[...,0], comp[...,1])  # [M, F, T]
                    out_wav = self.istft[str(fs_src)](out, cplx=True, squeeze=False) # B, F, T -> B, L
                    out_wav_16k = self.resampler(out_wav)
                    out_wav_norm = self.normalize(out_wav_16k)
                    wav_len_list.append(out_wav.shape[-1])
                    wav_len_list_16k.append(out_wav_16k.shape[-1])
                    
                if target is not None:
                    src_wav = target.to(self.device)
                    # src_wav = self.istft[str(fs_src)](target_stft, cplx=True) # B, F, T -> B, L
                    src_wav_16k = self.resampler(src_wav)
                    src_wav_norm = self.normalize(src_wav_16k)
                    wav_len_list.append(src_wav.shape[-1])
                    wav_len_list_16k.append(src_wav_16k.shape[-1])


                min_len = min(wav_len_list)
                min_len_16k = min(wav_len_list_16k)
                if self.input_eval:
                    in_wav = in_wav[...,:min_len]
                    in_wav_16k = in_wav_16k[...,:min_len_16k]
                    in_wav_norm = in_wav_norm[...,:min_len_16k]
                if self.output_eval:
                    out_wav = out_wav[...,:min_len]
                    out_wav_16k = out_wav_16k[...,:min_len_16k]
                    out_wav_norm = out_wav_norm[...,:min_len_16k]
                if target is not None:
                    src_wav = src_wav[...,:min_len]
                    src_wav_16k = src_wav_16k[...,:min_len_16k]
                    src_wav_norm = src_wav_norm[...,:min_len_16k]

                
                # ------------- Non-intrusive MOS metrics  ----------- #
                if 'wvmos' in self.metrics_key:
                    if target is not None:
                        cur_wvmos_src = self.wvmos(src_wav_norm)
                        self._update_metrics(metrics, 'wvmos_src', None, cur_wvmos_src.item())
                    cur_wvmos_in = self.wvmos(in_wav_norm).item() if self.input_eval else None
                    cur_wvmos = self.wvmos(out_wav_norm).item() if self.output_eval else None
                    self._update_metrics(metrics, 'wvmos', cur_wvmos_in, cur_wvmos)

                if 'utmos' in self.metrics_key:
                    if target is not None:
                        cur_utmos_src = self.utmos.mos(src_wav_norm)
                        self._update_metrics(metrics, 'utmos_src', None, cur_utmos_src.item())
                    cur_utmos_in = self.utmos.mos(in_wav_norm).item() if self.input_eval else None
                    cur_utmos = self.utmos.mos(out_wav_norm).item() if self.output_eval else None
                    self._update_metrics(metrics, 'utmos', cur_utmos_in, cur_utmos)

                if 'dnsmos' in self.metrics_key:
                    if target is not None:
                        cur_dnsmos_src = util_dnsmos.run(src_wav_norm.squeeze().cpu().numpy(), 16000)
                        self._update_metrics(metrics, 'dnsmos_src', None, cur_dnsmos_src)
                    cur_dnsmos_in = util_dnsmos.run(in_wav_norm.squeeze().cpu().numpy(), 16000)['ovrl_mos'] if self.input_eval else None
                    cur_dnsmos = util_dnsmos.run(out_wav_norm.squeeze().cpu().numpy(), 16000)['ovrl_mos'] if self.output_eval else None
                    self._update_metrics(metrics, 'dnsmos', cur_dnsmos_in, cur_dnsmos)

                # ------------- Intrusive Metrics ----------- #
                if target is not None:
                    # ----------------- Wideband (16kHz) metrics ----------------- #
                    if 'pesq' in self.metrics_key:
                        cur_pesq_wb_in = util_metric.run_metric(in_wav_16k, src_wav_16k, 'PESQ', 16000) if self.input_eval else None
                        cur_pesq_wb = util_metric.run_metric(out_wav_16k, src_wav_16k, 'PESQ', 16000) if self.output_eval else None
                        self._update_metrics(metrics, 'pesq', cur_pesq_wb_in, cur_pesq_wb)
                    if 'stoi' in self.metrics_key:
                        cur_stoi_in = util_metric.run_metric(in_wav_16k, src_wav_16k, 'STOI', 16000) if self.input_eval else None
                        cur_stoi = util_metric.run_metric(out_wav_16k, src_wav_16k, 'STOI', 16000) if self.output_eval else None
                        self._update_metrics(metrics, 'stoi', cur_stoi_in, cur_stoi)

                    # ----------------- Fullband (upto 48kHz) metrics ----------------- #
                    if 'lsd' in self.metrics_key:
                        cur_lsd_in = util_metric.run_metric(in_wav, src_wav, 'LSD', self.fs_src) if self.input_eval else None
                        cur_lsd = util_metric.run_metric(out_wav, src_wav, 'LSD', self.fs_src) if self.output_eval else None
                        self._update_metrics(metrics, 'lsd', cur_lsd_in, cur_lsd)
                    if 'sdr' in self.metrics_key:
                        cur_sdr_in = util_metric.run_metric(in_wav, src_wav, 'SNR', self.fs_src).item() if self.input_eval else None
                        cur_sdr = util_metric.run_metric(out_wav, src_wav, 'SNR', self.fs_src).item() if self.output_eval else None
                        self._update_metrics(metrics, 'sdr', cur_sdr_in, cur_sdr)
                    if 'mcd' in self.metrics_key:
                        cur_mcd_in = util_metric.run_metric(in_wav, src_wav, 'MCD', self.fs_src) if self.input_eval else None
                        cur_mcd = util_metric.run_metric(out_wav, src_wav, 'MCD', self.fs_src) if self.output_eval else None
                        self._update_metrics(metrics, 'mcd', cur_mcd_in, cur_mcd)

                    # ----------------- Downstream dependent metrics (16kHz Only) ----------------- #
                    #  BLEU, BERTscore, Token distance
                    if 'bleu' in self.metrics_key:
                        bleu_in = self.bleu_scorer.score(src_wav_16k, in_wav_16k) if self.input_eval else None
                        bleu_out = self.bleu_scorer.score(src_wav_16k, out_wav_16k) if self.output_eval else None
                        self._update_metrics(metrics, 'bleu', bleu_in, bleu_out)
                    if 'bertscore' in self.metrics_key:
                        bert_in, _, _ = self.bert_scorer.score(src_wav_16k, in_wav_16k) if self.input_eval else (None, None, None)
                        bert_out, _, _ = self.bert_scorer.score(src_wav_16k, out_wav_16k) if self.output_eval else (None, None, None)
                        self._update_metrics(metrics, 'bertscore', bert_in, bert_out)
                    if 'tokendist' in self.metrics_key:
                        tokendist_in = self.token_dist_scorer.score(src_wav_16k, in_wav_16k) if self.input_eval else None
                        tokendist_out = self.token_dist_scorer.score(src_wav_16k, out_wav_16k) if self.output_eval else None
                        self._update_metrics(metrics, 'tokendist', tokendist_in, tokendist_out)

                # save randomly chosen sample to writer
                if self.output_eval and self.input_eval:
                    if n_batch in self.random_sample_idx:
                        self.logging_sample_wav(out_wav, in_wav, target, epoch, n_batch)

                # update the progress bar
                n_batch += 1
                pbar.update(1)
                
                #! --------------- metrics for output ---------------- !#
                if (target is not None) and self.output_eval:
                    metrics['loss_se'] += self.loss(out_wav, src_wav, epoch=epoch).item()
                    metrics['loss_time'] += self.loss_t(out_wav, src_wav).item()
                    metrics['loss_rep'] += self.loss_fm(out_wav, src_wav).item()
                    # --- Calculate mean and confidence for display ---
                    dict_loss = {
                        'L_time': metrics['loss_time']/n_batch,
                        'L_se': metrics['loss_se']/n_batch,
                        'L_rep': metrics['loss_rep']/n_batch,
                    }
                else:
                    dict_loss = {
                        'L_time': 0.0,
                        'L_se': 0.0,
                        'L_rep': 0.0,
                    }
                
                suffixes = ['', '_i', '_src']
                dict_metric_mean = {}
                for key in self.metrics_key:
                    for suffix in suffixes:
                        if key + suffix in metrics:
                            dict_metric_mean[key + suffix] = metrics[key + suffix] / n_batch
                
                def cfd_95(x, x_2, n):
                    if n == 0: return 0.0
                    std = (x_2 / n - (x / n) ** 2) ** 0.5
                    return 1.96 * std / (n ** 0.5)

                dict_metric_cfd = {
                    k: cfd_95(metrics[k], metrics[k+'_2'], n_batch) 
                    for k in dict_metric_mean if k+'_2' in metrics
                }
                # update pbar
                if self.input_eval and self.output_eval:
                    pbar.set_postfix({
                        k: f"{dict_metric_mean[k+'_i']:.2f}>>{dict_metric_mean[k]:.2f}"
                        for k in dict_metric_mean if '_src' not in k if '_i' not in k if '_2' not in k
                    })
                elif self.input_eval:
                    pbar.set_postfix({
                        k: f"{dict_metric_mean[k]:.2f}"
                        for k in dict_metric_mean if '_src' not in k if '_2' not in k
                    })                    
                elif self.output_eval:
                    pbar.set_postfix({
                        k: f"{dict_metric_mean[k]:.2f}"
                        for k in dict_metric_mean if '_src' not in k if '_i' not in k if '_2' not in k
                    })

            pbar.close()

            return dict_loss, dict_metric_mean, dict_metric_cfd, n_batch


    def logging_sample_wav(self, estim_src, noisy, target, epoch, idx):

        self.writer_src.log_wav2spec(noisy.squeeze(0), "test_noisy_distort_"+str(idx), epoch)
        self.writer_src.log_audio(noisy.squeeze(0), 'test_noisy_distort_audio_'+str(idx), epoch)

        self.writer_src.log_wav2spec(estim_src.squeeze(0), "test_enhance_out_"+str(idx), epoch)
        self.writer_src.log_audio(estim_src.squeeze(0), 'test_estim_enhance_audio_'+str(idx), epoch)

        if target is not None:
            self.writer_src.log_wav2spec(target.squeeze(0), "test_clean_"+str(idx), epoch)
            self.writer_src.log_audio(target.squeeze(0), 'test_clean_audio_'+str(idx), epoch)


    @logger_wraps()
    def run_eval(self):
        with torch.cuda.device(self.device):
            loss_dict, metric_mean, metric_cfd, n_batch = self._evaluate(self.dataloaders['test'], self.start_epoch-1)
            
            # --- Logging to terminal ---
            def log_metrics(log_title, metrics_dict):
                log_str = f"[{log_title}] Metrics(time/mini-batch)\n Epoch {self.start_epoch-1:2d}: "
                # Define the order of metrics to be logged
                suffixes = ['', '_i', '_src']
                
                log_parts = []
                for key in self.metrics_key:
                    for suffix in suffixes:
                        if key + suffix in metrics_dict:
                            log_parts.append(f"{key + suffix}: {metrics_dict[key + suffix]:.3f}")

                log_str += " | ".join(log_parts)
                logger.info(log_str)

            log_metrics("MEAN", metric_mean)
            log_metrics("95% CONFIDENCE SCORE", metric_cfd)
            
            # --- Logging to tensorboard ---
            if self.config['dataset_test']['tensorboard_logging']:
                # Log loss values
                loss_log_dict = {'STFT_L1': loss_dict['L_se'], 'Time_L1': loss_dict['L_time'], 'SSL_rep': loss_dict['L_rep']}
                self.writer_src.add_scalars(f"Loss_Test_{self.train_phase}", loss_log_dict, self.start_epoch)
                
                # Log mean metrics
                # Filter out keys that are not needed for tensorboard logging
                mean_log_dict = {k: v for k, v in metric_mean.items() if 'utt_' not in k}
                self.writer_src.add_scalars(f"Metrics_Test_Mean_{self.train_phase}", mean_log_dict, self.start_epoch)
                
                # Log confidence scores
                cfd_log_dict = {k: v for k, v in metric_cfd.items() if 'utt_' not in k}
                self.writer_src.add_scalars(f"Metrics_Test_cfd_95_{self.train_phase}", cfd_log_dict, self.start_epoch)

                self.writer_src.flush()

        logger.info(f"Evaluation for {self.config['dataset_test']['testset_key']} dataset done!")
        
        # Close the writer to properly save all data and release resources
        self.writer_src.close()
        logger.info("TensorBoard writer closed successfully")


    @logger_wraps()
    def run_infer(self):
        with torch.cuda.device(self.device):
            loss_dict, metric_mean, metric_cfd, n_batch = self._inference(self.dataloaders['test'], self.start_epoch-1)
            
        logger.info(f"Inference and file writing for {self.config['dataset_test']['testset_key']} dataset done!")
        
        # Close the writer to properly save all data and release resources
        self.writer_src.close()
        logger.info("TensorBoard writer closed successfully")
        



class EngineInfer(object):
    def __init__(self, args, config, model, dataloaders, gpuid, device):

        ''' Default setting '''
        self.args = args  # Store args for later use
        self.engine_mode = args.engine_mode
        self.config = config
        self.loader_config = config["dataloader"]
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        self.subset_conf = {}
        self.subset_conf["train"] = config["engine"]["subset"]["train"]
        self.subset_conf["valid"] = config["engine"]["subset"]["valid"]
        testset_key = self.config['dataset_test']['testset_key']
        self.fs_src = config['dataset_test'][testset_key]['sample_rate_src']
        self.fs_in = config['dataset_test'][testset_key]['sample_rate_in']

        # optim, scheduler, STFT configuration
        optim_cls = getattr(torch.optim, self.config["engine"]["optimizer"]["name"])
        self.stft, self.istft = {}, {}
        self.fs_list = config['fs_list']
        for fs in self.fs_list:
            frame_len = int(config['stft']['frame_length'] * int(fs) / 1000)
            frame_hop = int(config['stft']['frame_shift'] * int(fs) / 1000)
            self.stft[fs] = util_stft.STFT(frame_len, frame_hop, device=self.device, normalize=True)
            self.istft[fs] = util_stft.iSTFT(frame_len, frame_hop, device=self.device, normalize=True)
        self.out_F = int(config['stft']['frame_length'] * int(self.fs_src) / 1000) // 2 + 1

        # load enhance model
        self.train_phase = config["train_phase"] + '_' + config["dataset_phase"]
        config_name = self.args.config if hasattr(self.args, 'config') else 'default'
        log_base = f"log/log_{self.train_phase}_{config_name}"
        self.main_optimizer = optim_cls(self.model.parameters(),
                                        **self.config["engine"]["optimizer"].get(self.config["engine"]["optimizer"]["name"], {}))
        self.chkp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_base, "weights")
        os.makedirs(self.chkp_path, exist_ok=True)
        assert os.path.exists(self.chkp_path), f"Checkpoint path {self.chkp_path} does not exist!"
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.chkp_path, self.model, self.main_optimizer, location=self.device)
        
        if self.args.dump_path is not None:
            self.inference_dump_path = os.path.join(self.args.dump_path, "inference_wav", 
                                        f"{testset_key}_{self.train_phase}_{config_name[:-5]}_{str(self.fs_in)[:-3]}kto{str(self.fs_src)[:-3]}k")
            self.input_dump_path = os.path.join(self.args.dump_path, "inference_wav", 
                                        f"{testset_key}_input_{str(self.fs_in)[:-3]}kto{str(self.fs_src)[:-3]}k")
        else:
            self.inference_dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_wav", 
                                        f"{testset_key}_{self.train_phase}_{config_name[:-5]}_{str(self.fs_in)[:-3]}kto{str(self.fs_src)[:-3]}k")
            self.input_dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_wav", 
                                        f"{testset_key}_input_{str(self.fs_in)[:-3]}kto{str(self.fs_src)[:-3]}k")
        os.makedirs(self.inference_dump_path, exist_ok=True)
        logger.info(f"Inference dump path: {self.inference_dump_path}")

        
    @logger_wraps()
    def _inference(self, dataloader):
        self.model.eval()
        
        pbar = tqdm(total=len(dataloader), unit='utt', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="WHITE", dynamic_ncols=True)
        with torch.inference_mode():
            for batch in dataloader:
                noisy_orig = batch["noisy_distort_input"].type(torch.float32)
                noisy = batch["noisy_distort"].type(torch.float32)
                file_name = batch["file_name"]
                fs_in = batch["fs_in"].item()
                fs_src = batch["fs_src"].item()
                # feature pre-processing

                noisy_orig_stft = self.stft[str(fs_in)](noisy_orig.to(self.device), cplx=True)  # B, F, T

                # generator
                model_input = torch.stack([torch.real(noisy_orig_stft), torch.imag(noisy_orig_stft)],dim=-1) # B, F, T, 2
                comp = self.model(model_input, out_F=self.out_F) # B, F, T, 2
                out = torch.complex(comp[...,0], comp[...,1])  # [M, F, T]
                out_wav = self.istft[str(fs_src)](out, cplx=True, squeeze=False) # B, F, T -> B, L
                
                # write the enhanced wav file
                sf.write(os.path.join(self.inference_dump_path, f"{file_name[0]}.wav"), out_wav[0].cpu().numpy(), fs_src)
                sf.write(os.path.join(self.input_dump_path, f"{file_name[0]}.wav"), noisy[0].cpu().numpy(), fs_src)

                pbar.update(1)
            pbar.close()


    @logger_wraps()
    def run_infer(self, sample_file=None):
        with torch.cuda.device(self.device):
            self._inference(self.dataloaders['test'])
            
        logger.info(f"Inference and file writing for {self.config['dataset_test']['testset_key']} dataset done!")
        
        
