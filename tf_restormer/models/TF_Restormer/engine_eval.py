from __future__ import annotations

import argparse
import os
import torch
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tf_restormer.utils import util_engine, util_stft, util_writer
from tf_restormer.utils.util_engine import resolve_log_base
from tf_restormer.utils.metrics import compute_metric
from tf_restormer.utils.decorators import logger_wraps


# @logger_wraps()
class EngineEval(object):
    """Evaluation engine computing comprehensive metrics on test sets."""

    def __init__(self, args: argparse.Namespace, config: dict, model: torch.nn.Module, dataloaders: dict, gpuid: tuple, device: torch.device) -> None:

        ''' Default setting '''
        self.args = args  # Store args for later use
        self.engine_mode = args.engine_mode
        self.config = config
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        testset_key = self.config['dataset_test']['testset_key']
        self.input_eval = config["dataset_test"].get('input_eval', True)
        self.output_eval = config["dataset_test"].get('output_eval', True)
        self.fs_src = config['dataset_test'][testset_key]['sample_rate_src']
        self.fs_in = config['dataset_test'][testset_key]['sample_rate_in']
        self.metrics_key = config['dataset_test'][testset_key].get('metrics', [])
        self.resampler = T.Resample(orig_freq=self.fs_src, new_freq=16000,
                                    resampling_method='sinc_interp_hann',
                                    lowpass_filter_width=32,
                                    rolloff=0.98).to(self.device)
        # loss configuration (lazy import — optional train extras)
        self.train_phase = config["train_phase"]
        self._loss_modules_available = False
        try:
            from .loss import SSL_FM_Loss, MS_STFT_Gen_SC_Loss, Time_Domain_L1
            self.w = config["engine"][self.train_phase]["loss_weight"]
            self.loss_t = Time_Domain_L1(**config["engine"][self.train_phase]["loss_time"], device=self.device)
            self.loss = MS_STFT_Gen_SC_Loss(**config["engine"][self.train_phase]["loss_enhance"], device=self.device)
            self.loss_fm = SSL_FM_Loss(**config["engine"][self.train_phase]["loss_rep"], device=self.device)
            self._loss_modules_available = True
        except ImportError as e:
            logger.warning(f"Loss modules not available: {e}. Loss computation will be skipped.")
        except KeyError as e:
            logger.warning(f"Loss config key missing: {e}. Loss computation will be skipped (eval-only config?).")

        # STFT configuration
        self.stft, self.istft = {}, {}
        self.fs_list = config['fs_list']
        for fs in self.fs_list:
            frame_len = int(config['stft']['frame_length'] * int(fs) / 1000)
            frame_hop = int(config['stft']['frame_shift'] * int(fs) / 1000)
            self.stft[fs] = util_stft.STFT(frame_len, frame_hop, device=self.device, normalize=True)
            self.istft[fs] = util_stft.iSTFT(frame_len, frame_hop, device=self.device, normalize=True)
        self.out_F = int(config['stft']['frame_length'] * int(self.fs_src) / 1000) // 2 + 1

        self.sample_file_list = config['engine'].get('sample_validation', [])
        # load enhance model (model weights only — no optimizer state needed for eval)
        config_name = self.args.config if hasattr(self.args, 'config') else 'default'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_base = resolve_log_base(self.train_phase, config_name, base_dir)
        self.chkp_path = os.path.join(base_dir, log_base, "weights")
        os.makedirs(self.chkp_path, exist_ok=True)
        assert os.path.exists(self.chkp_path), f"Checkpoint path {self.chkp_path} does not exist!"
        util_engine.load_last_checkpoint_n_get_epoch_model_only(self.chkp_path, self.model, location=self.device)
        self.start_epoch = 1  # epoch is not tracked in eval mode; used only for logging

        self.audio_log_path = os.path.join(base_dir, log_base, "tensorboard_eval_"+testset_key)
        os.makedirs(self.audio_log_path, exist_ok=True)
        self.writer_src = util_writer.TBWriter(logdir=self.audio_log_path,
                                               n_fft=config['stft']['frame_length'] * self.fs_src // 16000,
                                               n_hop=config['stft']['frame_shift'] * self.fs_src // 16000,
                                               sr=self.fs_src)

        self.random_sample_idx = self.config['dataset_test'][testset_key].get('random_sample_idx', [10, 20, 30, 40, 50])

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
            if value_in is not None:
                metrics[f'{key}_i'] += value_in
                metrics[f'{key}_i_2'] += value_in**2
            if value_out is not None:
                metrics[f'{key}'] += value_out
                metrics[f'{key}_2'] += value_out**2

    @logger_wraps()
    def _evaluate(self, dataloader: DataLoader, epoch: int) -> tuple[dict, dict, dict, int]:
        self.model.eval()

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


                # ------------- Non-intrusive / neural MOS metrics  ----------- #
                # wvmos, utmos, dnsmos, dnsmos_sig, dnsmos_bak, nisqa
                # Use z-normalized 16kHz variants (in_wav_norm, out_wav_norm, src_wav_norm)
                for key in ('wvmos', 'utmos', 'dnsmos', 'dnsmos_sig', 'dnsmos_bak', 'nisqa'):
                    if key in self.metrics_key:
                        if target is not None:
                            # [BUGFIX] _src values bypass _update_metrics (wvmos_src etc.
                            # are not in self.metrics_key, so _update_metrics would silently drop them)
                            val_src = compute_metric(key, src_wav_norm, fs=16000, device=self.device)
                            val_src = val_src.item() if hasattr(val_src, 'item') else float(val_src)
                            metrics[f'{key}_src'] += val_src
                            metrics[f'{key}_src_2'] += val_src ** 2
                        val_in = compute_metric(key, in_wav_norm, fs=16000, device=self.device) if self.input_eval else None
                        if val_in is not None:
                            val_in = val_in.item() if hasattr(val_in, 'item') else float(val_in)
                        val_out = compute_metric(key, out_wav_norm, fs=16000, device=self.device) if self.output_eval else None
                        if val_out is not None:
                            val_out = val_out.item() if hasattr(val_out, 'item') else float(val_out)
                        self._update_metrics(metrics, key, val_in, val_out)

                # ------------- Intrusive Metrics ----------- #
                if target is not None:
                    # ----------------- Wideband (16kHz) metrics ----------------- #
                    for key in ('pesq', 'stoi'):
                        if key in self.metrics_key:
                            val_in = compute_metric(key, in_wav_16k, src_wav_16k, 16000) if self.input_eval else None
                            val_out = compute_metric(key, out_wav_16k, src_wav_16k, 16000) if self.output_eval else None
                            self._update_metrics(metrics, key, val_in, val_out)

                    # ----------------- Fullband (upto 48kHz) metrics ----------------- #
                    for key in ('lsd', 'sdr', 'mcd'):
                        if key in self.metrics_key:
                            val_in = compute_metric(key, in_wav, src_wav, self.fs_src) if self.input_eval else None
                            val_out = compute_metric(key, out_wav, src_wav, self.fs_src) if self.output_eval else None
                            # SDR returns tensor, need .item()
                            if val_in is not None and hasattr(val_in, 'item'):
                                val_in = val_in.item()
                            if val_out is not None and hasattr(val_out, 'item'):
                                val_out = val_out.item()
                            self._update_metrics(metrics, key, val_in, val_out)

                    # ----------------- Downstream dependent metrics (16kHz Only) ----------------- #
                    #  BLEU, BERTscore, Token distance — pass device kwarg for GPU model loading
                    if 'bleu' in self.metrics_key:
                        bleu_in = compute_metric('bleu', in_wav_16k, src_wav_16k, 16000, device=self.device) if self.input_eval else None
                        bleu_out = compute_metric('bleu', out_wav_16k, src_wav_16k, 16000, device=self.device) if self.output_eval else None
                        self._update_metrics(metrics, 'bleu', bleu_in, bleu_out)
                    if 'bertscore' in self.metrics_key:
                        bert_in_result = compute_metric('bertscore', in_wav_16k, src_wav_16k, 16000, device=self.device) if self.input_eval else (None, None, None)
                        bert_out_result = compute_metric('bertscore', out_wav_16k, src_wav_16k, 16000, device=self.device) if self.output_eval else (None, None, None)
                        bert_in = bert_in_result[0] if isinstance(bert_in_result, tuple) else bert_in_result
                        bert_out = bert_out_result[0] if isinstance(bert_out_result, tuple) else bert_out_result
                        self._update_metrics(metrics, 'bertscore', bert_in, bert_out)
                    if 'tokendist' in self.metrics_key:
                        tokendist_in = compute_metric('tokendist', in_wav_16k, src_wav_16k, 16000, device=self.device) if self.input_eval else None
                        tokendist_out = compute_metric('tokendist', out_wav_16k, src_wav_16k, 16000, device=self.device) if self.output_eval else None
                        self._update_metrics(metrics, 'tokendist', tokendist_in, tokendist_out)

                    # ----------------- ASR-based error rate metrics (16kHz Only) ----------------- #
                    #  wer_whisper, wer_w2v, cer_whisper — return (err, ref_len) tuples for accumulation
                    for key in ('wer_whisper', 'wer_w2v', 'cer_whisper'):
                        if key in self.metrics_key:
                            if self.input_eval:
                                res_in = compute_metric(key, in_wav_16k, src_wav_16k, 16000, device=self.device)
                                metrics[f'{key}_i_err'] += res_in[0]
                                metrics[f'{key}_i_ref'] += res_in[1]
                            if self.output_eval:
                                res_out = compute_metric(key, out_wav_16k, src_wav_16k, 16000, device=self.device)
                                metrics[f'{key}_err'] += res_out[0]
                                metrics[f'{key}_ref'] += res_out[1]

                # save randomly chosen sample to writer
                if self.output_eval and self.input_eval:
                    if n_batch in self.random_sample_idx:
                        self.logging_sample_wav(out_wav, in_wav, target, epoch, n_batch)

                # update the progress bar
                n_batch += 1
                pbar.update(1)

                #! --------------- metrics for output ---------------- !#
                if (target is not None) and self.output_eval and self._loss_modules_available:
                    metrics['loss_se'] += self.loss(out_wav, src_wav, epoch=epoch).item()
                    metrics['loss_time'] += self.loss_t(out_wav, src_wav).item()
                    metrics['loss_rep'] += self.loss_fm(out_wav, src_wav).item()

                # --- Calculate mean and confidence for display ---
                if (target is not None) and self.output_eval and self._loss_modules_available:
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

                # ASR metrics are accumulated as (err, ref_len) pairs.
                # Compute rate = sum(err) / sum(ref_len) and add to dict_metric_mean.
                for asr_key in ('wer_whisper', 'wer_w2v', 'cer_whisper'):
                    if asr_key in self.metrics_key:
                        for err_sfx, ref_sfx, out_sfx in [
                            ('_err',   '_ref',   ''),
                            ('_i_err', '_i_ref', '_i'),
                        ]:
                            err_k = asr_key + err_sfx
                            ref_k = asr_key + ref_sfx
                            if err_k in metrics and metrics[ref_k] > 0:
                                dict_metric_mean[asr_key + out_sfx] = (
                                    metrics[err_k] / metrics[ref_k]
                                )

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
    def run_eval(self) -> None:
        """Run evaluation on all configured test sets."""
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
            if self.config['dataset_test'].get('tensorboard_logging', True):
                # Log loss values (only when loss modules were available)
                if self._loss_modules_available:
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
