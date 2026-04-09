from __future__ import annotations

import argparse
import os
import torch
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tf_restormer.utils import util_engine, util_stft, util_metric, util_writer, util_wvmos, util_dnsmos, util_sBERTscore
from tf_restormer.utils.decorators import logger_wraps
from .loss import SSL_FM_Loss, MS_STFT_Gen_SC_Loss, Time_Domain_L1, UTMOS_Loss


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

        self.sample_file_list = config['engine'].get('sample_validation', [])
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
            if value_in != None:
                metrics[f'{key}_i'] += value_in
                metrics[f'{key}_i_2'] += value_in**2
            if value_out != None:
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
