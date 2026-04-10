from __future__ import annotations

import argparse
import os
import torch
import time
import random
from loguru import logger
from tqdm import tqdm
import librosa
import torchaudio.transforms as T
from torchaudio.functional import resample as torch_resample
from torchaudio.io import AudioEffector, CodecConfig
from torch.utils.data import DataLoader
from tf_restormer.utils import util_engine, util_stft, util_writer
from tf_restormer.utils.decorators import logger_wraps
from .loss import SSL_FM_Loss, MS_STFT_Gen_SC_Loss, Time_Domain_L1, HF_Loss
from .modules.msstftd import SFIMultiScaleSTFTDiscriminator

# @logger_wraps()
class Engine(object):
    """Training engine with two-stage (pretrain/adversarial) support."""

    def __init__(self, args: argparse.Namespace, config: dict, model: torch.nn.Module, dataloaders: dict, gpuid: tuple, device: torch.device) -> None:

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
        self.writer_src = util_writer.TBWriter(logdir=self.audio_log_path,
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


    def audio_effecter(self, audio: torch.Tensor, sample_rate: int, batch_wise: bool = True) -> tuple[torch.Tensor, int]:
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

    def target_downsample(self, target: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if random.random() < self.prob_downsample_src:
            fs_target = int(random.choice(self.fs_list_src))
            target = torch_resample(target, orig_freq=self.fs_src, new_freq=fs_target, lowpass_filter_width=32, rolloff=0.98)
            out_F = int(self.config['stft']['frame_length'] * fs_target / 1000) // 2 + 1
        else:
            fs_target = self.fs_src
            out_F = self.out_F

        return target, fs_target, out_F

    def _discriminator_step(self, out: torch.Tensor, src: torch.Tensor, fs: int, update: bool) -> torch.Tensor:
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


    def _generator_step(self, out: torch.Tensor, src: torch.Tensor, fs: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    def _train(self, _dataloader: DataLoader, epoch: int) -> tuple[dict, int]:
        """Run one training epoch."""
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
    def _validate(self, _dataloader: DataLoader, epoch: int) -> tuple[dict, int]:
        """Run one validation epoch."""
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



    @logger_wraps()
    def run(self) -> None:
        """Execute full training loop with validation and checkpointing."""
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
