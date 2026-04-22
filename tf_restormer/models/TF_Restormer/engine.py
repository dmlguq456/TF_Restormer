from __future__ import annotations

import argparse
import os
import random
import time

import torch
from loguru import logger
from tqdm import tqdm
import librosa
import torchaudio.transforms as T
from torchaudio.functional import resample as torch_resample
from torchaudio.io import AudioEffector, CodecConfig
from torch.utils.data import DataLoader
from tf_restormer.utils import util_engine, util_stft
from tf_restormer.utils.util_engine import resolve_log_base
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
        self.fs_src = config["dataset"]["sample_rate_src"]
        self.fs_in = config["dataset"]["sample_rate_in"]

        self.train_phase = config["train_phase"]
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
        self.main_optimizer, self.warmup_scheduler, self.main_scheduler, optim_cls, sched_cls = \
            util_engine.setup_optimizer_and_scheduler(self.model, config)

        # load enhance model
        config_name = self.args.config if hasattr(self.args, 'config') else 'default'
        self.config_name = config_name  # store for format_epoch_log

        self.chkp_path, self.writer_src, self.start_epoch = util_engine.setup_logging(
            config_name, self.train_phase,
            os.path.dirname(os.path.abspath(__file__)),
            self.model, self.main_optimizer, self.device,
            config=config, fs_src=self.fs_src)

        train_phase_list = config['train_phase_list']
        if 'adversarial' in self.train_phase:
            # Fallback from pretrain if no fine-tune checkpoints exist
            files = sorted(os.listdir(self.chkp_path))
            if not files:
                # locate pretrain checkpoint directory
                prev_stage = train_phase_list[train_phase_list.index(self.train_phase) - 1]

                base_dir = os.path.dirname(os.path.abspath(__file__))
                prev_log_base = resolve_log_base(prev_stage, config_name, base_dir)
                pretrain_dir = os.path.join(base_dir, prev_log_base, "weights")
                pre_files = sorted([f for f in os.listdir(pretrain_dir) if f.endswith('.pth')])
                if pre_files:
                    last_ckpt = pre_files[-1]
                    _ = util_engine.load_last_checkpoint_n_get_epoch(pretrain_dir, self.model, self.main_optimizer, location=self.device)
                    util_engine.save_checkpoint_per_nth(1, 0, self.model, self.main_optimizer, 1.0e5,  1.0e5, self.chkp_path)
                    logger.info(f"Copied pretrain checkpoint {last_ckpt} to fine-tune as epoch.0000.pth")
                # load or fallback checkpoint
                self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.chkp_path, self.model, self.main_optimizer, location=self.device)

        # Step 2.3: initialize BestModelTracker for Generator
        self.best_tracker = util_engine.BestModelTracker(mode="min")
        self.best_tracker.restore(self.chkp_path)

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
            self.chkp_path_D = os.path.join(os.path.dirname(self.chkp_path), "weights_D")
            os.makedirs(self.chkp_path_D, exist_ok=True)
            # Note: start_epoch is overwritten here with D's epoch.
            # G/D epoch mismatch: training resumes from D's last epoch (legacy behavior).
            self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.chkp_path_D, self.msstftd, self.optimizer_disc, location=self.device)
            # Step 2.3: initialize BestModelTracker for Discriminator
            # D uses single-metric L_D_adv tracking (not a composite loss)
            self.best_tracker_D = util_engine.BestModelTracker(mode="min")
            self.best_tracker_D.restore(self.chkp_path_D)

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
        dataloader = util_engine.create_dataloader_with_sampler(
            _dataloader, self.subset_conf["train"], self.loader_config
        )

        self.model.train()
        tot_loss_time, tot_loss_se, tot_loss_rep, tot_loss_pesq, tot_loss_G_fm, tot_loss_G_adv, tot_loss_disc, n_batch = 0, 0, 0, 0, 0, 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batch', desc='TRAIN', colour="YELLOW", dynamic_ncols=True, bar_format=util_engine.PBAR_FMT)
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
            pbar.set_postfix(util_engine.format_pbar(dict_loss))
        pbar.close()

        return dict_loss, n_batch

    @logger_wraps()
    def _validate(self, _dataloader: DataLoader, epoch: int) -> tuple[dict, int]:
        """Run one validation epoch."""
        valid_conf = dict(self.subset_conf["valid"])  # shallow copy
        if valid_conf["subset"] and epoch == 0:
            valid_conf["num_per_epoch"] = 100  # epoch 0 sanity check
        dataloader = util_engine.create_dataloader_with_sampler(
            _dataloader, valid_conf, self.loader_config,
        )

        self.model.eval()
        tot_loss_time, tot_loss_se, tot_loss_rep, tot_loss_pesq, tot_loss_G_fm, tot_loss_G_adv, tot_loss_disc, n_batch = 0, 0, 0, 0, 0, 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batch', desc='VALID', colour="RED", dynamic_ncols=True, bar_format=util_engine.PBAR_FMT)

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
                pbar.set_postfix(util_engine.format_pbar(dict_loss))
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
            init_start_time = time.time()
            init_loss_dict, _ = self._validate(self.dataloaders['valid'], self.start_epoch-1)
            init_elapsed = time.time() - init_start_time
            logger.info(util_engine.format_epoch_log(
                self.config_name, self.start_epoch - 1, "INIT", init_loss_dict, init_elapsed))

            for epoch in range(self.start_epoch, self.config['engine']['max_epoch'][self.train_phase] + 1):

                # Training
                train_start_time = time.time()
                tr_loss_dict, tr_n_batch = self._train(self.dataloaders['train'], epoch)
                train_elapsed = time.time() - train_start_time

                # Validation #! validation is always based on end-to-end (not oracle enhance or restore)
                val_start_time   = time.time()
                val_loss_dict, val_n_batch = self._validate(self.dataloaders['valid'], epoch)
                val_elapsed = time.time() - val_start_time

                # Scheduling
                if epoch > self.config['engine']['start_scheduling'][self.train_phase]:
                    self.main_scheduler.step()
                    if 'adversarial' in self.train_phase:
                        self.scheduler_disc.step()

                # Logging to terminal (Step 2.4)
                logger.info(util_engine.format_epoch_log(
                    self.config_name, epoch, "TRAIN", tr_loss_dict, train_elapsed,
                    suffix=f" ({tr_n_batch:d} batches)"))
                logger.info(util_engine.format_epoch_log(
                    self.config_name, epoch, "VALID", val_loss_dict, val_elapsed,
                    suffix=f" ({val_n_batch:d} batches)"))

                if 'pretrain' in self.train_phase: # only when pretraining w/o adversarial training
                    val_loss_gen = (val_loss_dict['L_se']*self.w['se'] + val_loss_dict['L_rep']*self.w['ssl'])
                    tr_loss_gen = (tr_loss_dict['L_se']*self.w['se'] + tr_loss_dict['L_rep']*self.w['ssl'])
                else:
                    tr_loss_gen = (tr_loss_dict['L_se']*self.w['se'] + tr_loss_dict['L_rep']*self.w['ssl'] + tr_loss_dict['L_pesq']*self.w['pesq']
                                    + tr_loss_dict['L_G_adv']*self.w['gan'] + tr_loss_dict['L_G_fm']*self.w['fm'])
                    val_loss_gen = (val_loss_dict['L_se']*self.w['se'] + val_loss_dict['L_rep']*self.w['ssl'] + val_loss_dict['L_pesq']*self.w['pesq']
                                    + val_loss_dict['L_G_adv']*self.w['gan'] + val_loss_dict['L_G_fm']*self.w['fm'])

                # Save checkpoint (Step 2.3)
                # val_loss_gen is a weighted composite loss (pretrain: L_se + L_rep;
                # adversarial: 5-term sum). Composite tracking captures overall quality.
                util_engine.save_checkpoint_optimized(
                    epoch, self.model, self.main_optimizer, self.chkp_path,
                    val_metric=val_loss_gen, best_tracker=self.best_tracker,
                    train_loss=tr_loss_gen, valid_loss=val_loss_gen)
                if 'adversarial' in self.train_phase:
                    # L_D_adv is the sole D metric — single-metric tracking is correct here.
                    util_engine.save_checkpoint_optimized(
                        epoch, self.msstftd, self.optimizer_disc, self.chkp_path_D,
                        val_metric=val_loss_dict['L_D_adv'], best_tracker=self.best_tracker_D,
                        train_loss=tr_loss_dict['L_D_adv'], valid_loss=val_loss_dict['L_D_adv'])

                # Logging to TensorBoard (Step 2.5)
                # Use add_scalar (singular) with group/key format for cleaner TB grouping.
                # train_phase is now directly 'pretrain' or 'adversarial'; substring match
                # kept for forward compatibility with any future phase variants.
                tb_metrics = {
                    "Loss_se/train": tr_loss_dict['L_se'],
                    "Loss_se/valid": val_loss_dict['L_se'],
                    "Loss_time/train": tr_loss_dict['L_time'],
                    "Loss_time/valid": val_loss_dict['L_time'],
                    "Loss_rep/train": tr_loss_dict['L_rep'],
                    "Loss_rep/valid": val_loss_dict['L_rep'],
                    "LearningRate": self.main_optimizer.param_groups[0]['lr'],
                }
                if 'adversarial' in self.train_phase:
                    tb_metrics.update({
                        "Loss_G_adv/train": tr_loss_dict['L_G_adv'],
                        "Loss_G_adv/valid": val_loss_dict['L_G_adv'],
                        "Loss_G_fm/train": tr_loss_dict['L_G_fm'],
                        "Loss_G_fm/valid": val_loss_dict['L_G_fm'],
                        "Loss_D/train": tr_loss_dict['L_D_adv'],
                        "Loss_D/valid": val_loss_dict['L_D_adv'],
                        "Loss_pesq/train": tr_loss_dict['L_pesq'],
                        "Loss_pesq/valid": val_loss_dict['L_pesq'],
                    })
                util_engine.log_scalars_to_tb(self.writer_src, tb_metrics, epoch)

            logger.info(f"Training for {self.config['engine']['max_epoch']} epoches done!")

            # Close the writer to properly save all data and release resources
            self.writer_src.close()
            logger.info("TensorBoard writer closed successfully")
