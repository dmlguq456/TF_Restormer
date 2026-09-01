"""
Analyze correlation between target STFT magnitude and L1 distance (target vs output).
Supports GSC loss justification: adaptive weighting by source magnitude is principled,
not heuristic, because spectral errors are heteroscedastic (scale with source energy).

Uses SFI-STFT (40ms window, 10ms hop) consistent with the model.
"""

import os
import torch
import torchaudio
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict


def compute_stft(wav, n_fft, hop_length):
    """Compute STFT consistent with SFI-STFT (40ms frame)."""
    window = torch.hann_window(n_fft)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length,
                       win_length=n_fft, window=window,
                       return_complex=True, normalized=True)
    return stft  # (F, T)


def analyze_dataset(target_dir, output_dir, sr, dataset_name):
    """Analyze correlation at per-utterance and per-frequency-bin levels."""
    n_fft = int(0.040 * sr)   # 40ms frame
    hop_length = n_fft // 4   # 10ms hop

    target_files = set(os.listdir(target_dir))
    output_files = set(os.listdir(output_dir))
    common = sorted(target_files & output_files)

    if not common:
        print(f"[SKIP] {dataset_name}: no common files between target and output")
        return None

    # Per-utterance accumulators
    utt_target_rms = []
    utt_l1_mag = []
    utt_l1_real = []
    utt_l1_imag = []

    # Per-frequency-bin accumulators
    freq_bins = n_fft // 2 + 1
    freq_target_energy_sum = np.zeros(freq_bins)
    freq_l1_mag_sum = np.zeros(freq_bins)
    freq_l1_real_sum = np.zeros(freq_bins)
    freq_l1_imag_sum = np.zeros(freq_bins)

    # For normalized error uniformity
    freq_l1_mag_sq_sum = np.zeros(freq_bins)
    freq_target_sq_sum = np.zeros(freq_bins)

    n_files = 0
    skipped = 0

    for fname in common:
        if not fname.endswith('.wav'):
            continue
        try:
            tgt_wav, tgt_sr = torchaudio.load(os.path.join(target_dir, fname))
            out_wav, out_sr = torchaudio.load(os.path.join(output_dir, fname))
        except Exception as e:
            skipped += 1
            continue

        # Ensure mono
        tgt_wav = tgt_wav[0]
        out_wav = out_wav[0]

        # Align length
        min_len = min(tgt_wav.shape[-1], out_wav.shape[-1])
        if min_len < n_fft:
            skipped += 1
            continue
        tgt_wav = tgt_wav[:min_len]
        out_wav = out_wav[:min_len]

        # STFT
        tgt_stft = compute_stft(tgt_wav, n_fft, hop_length)  # (F, T)
        out_stft = compute_stft(out_wav, n_fft, hop_length)

        tgt_mag = torch.abs(tgt_stft)
        out_mag = torch.abs(out_stft)

        # === Per-utterance stats ===
        utt_rms = torch.sqrt(torch.mean(tgt_mag ** 2)).item()
        l1_mag = torch.mean(torch.abs(tgt_mag - out_mag)).item()
        l1_real = torch.mean(torch.abs(tgt_stft.real - out_stft.real)).item()
        l1_imag = torch.mean(torch.abs(tgt_stft.imag - out_stft.imag)).item()

        utt_target_rms.append(utt_rms)
        utt_l1_mag.append(l1_mag)
        utt_l1_real.append(l1_real)
        utt_l1_imag.append(l1_imag)

        # === Per-frequency-bin stats (average over time) ===
        freq_tgt_e = torch.mean(tgt_mag, dim=-1).numpy()       # (F,)
        freq_l1_m = torch.mean(torch.abs(tgt_mag - out_mag), dim=-1).numpy()
        freq_l1_r = torch.mean(torch.abs(tgt_stft.real - out_stft.real), dim=-1).numpy()
        freq_l1_i = torch.mean(torch.abs(tgt_stft.imag - out_stft.imag), dim=-1).numpy()

        freq_target_energy_sum += freq_tgt_e
        freq_l1_mag_sum += freq_l1_m
        freq_l1_real_sum += freq_l1_r
        freq_l1_imag_sum += freq_l1_i

        freq_target_sq_sum += freq_tgt_e ** 2
        freq_l1_mag_sq_sum += freq_l1_m ** 2

        n_files += 1

    if n_files == 0:
        print(f"[SKIP] {dataset_name}: no valid files processed")
        return None

    # Convert to numpy
    utt_target_rms = np.array(utt_target_rms)
    utt_l1_mag = np.array(utt_l1_mag)
    utt_l1_real = np.array(utt_l1_real)
    utt_l1_imag = np.array(utt_l1_imag)

    freq_target_e_avg = freq_target_energy_sum / n_files
    freq_l1_mag_avg = freq_l1_mag_sum / n_files
    freq_l1_real_avg = freq_l1_real_sum / n_files
    freq_l1_imag_avg = freq_l1_imag_sum / n_files

    # ============================================================
    # Print results
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Files: {n_files} (skipped: {skipped}), SR: {sr}Hz")
    print(f"  SFI-STFT: n_fft={n_fft}, hop={hop_length} (40ms/10ms)")
    print(f"  Freq bins: {freq_bins}")
    print(f"{'=' * 70}")

    # --- Per-Utterance ---
    print(f"\n  [Per-Utterance Correlation] (N={n_files})")
    print(f"  {'Metric':<12} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
    print(f"  {'-'*58}")
    for name, arr in [("L1_mag", utt_l1_mag), ("L1_real", utt_l1_real), ("L1_imag", utt_l1_imag)]:
        rp, pp = stats.pearsonr(utt_target_rms, arr)
        rs, ps = stats.spearmanr(utt_target_rms, arr)
        print(f"  {name:<12} {rp:>10.4f} {pp:>12.2e} {rs:>12.4f} {ps:>12.2e}")

    # --- Per-Frequency-Bin ---
    print(f"\n  [Per-Frequency-Bin Correlation] (N_bins={freq_bins})")
    print(f"  {'Metric':<12} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
    print(f"  {'-'*58}")
    for name, arr in [("L1_mag", freq_l1_mag_avg), ("L1_real", freq_l1_real_avg), ("L1_imag", freq_l1_imag_avg)]:
        rp, pp = stats.pearsonr(freq_target_e_avg, arr)
        rs, ps = stats.spearmanr(freq_target_e_avg, arr)
        print(f"  {name:<12} {rp:>10.4f} {pp:>12.2e} {rs:>12.4f} {ps:>12.2e}")

    # --- Normalized Error Uniformity ---
    print(f"\n  [Normalized Error Uniformity — per-freq bin]")
    eps = 1e-8
    raw_cv = np.std(freq_l1_mag_avg) / (np.mean(freq_l1_mag_avg) + eps)
    norm_err = freq_l1_mag_avg / (freq_target_e_avg + eps)
    norm_cv = np.std(norm_err) / (np.mean(norm_err) + eps)
    print(f"  Raw |error|          — CV (std/mean): {raw_cv:.4f}")
    print(f"  Normalized |err|/|s| — CV (std/mean): {norm_cv:.4f}")
    if raw_cv > eps:
        reduction = (1 - norm_cv / raw_cv) * 100
        print(f"  CV reduction by normalization: {reduction:+.1f}%")

    # --- Summary statistics ---
    print(f"\n  [Summary Stats]")
    print(f"  Target RMS  — mean: {np.mean(utt_target_rms):.6f}, std: {np.std(utt_target_rms):.6f}")
    print(f"  L1_mag      — mean: {np.mean(utt_l1_mag):.6f}, std: {np.std(utt_l1_mag):.6f}")
    print(f"  L1_real     — mean: {np.mean(utt_l1_real):.6f}, std: {np.std(utt_l1_real):.6f}")
    print(f"  L1_imag     — mean: {np.mean(utt_l1_imag):.6f}, std: {np.std(utt_l1_imag):.6f}")

    return {
        'dataset': dataset_name,
        'n_files': n_files,
        'utt_target_rms': utt_target_rms,
        'utt_l1_mag': utt_l1_mag,
        'utt_l1_real': utt_l1_real,
        'utt_l1_imag': utt_l1_imag,
        'freq_target_e_avg': freq_target_e_avg,
        'freq_l1_mag_avg': freq_l1_mag_avg,
        'freq_l1_real_avg': freq_l1_real_avg,
        'freq_l1_imag_avg': freq_l1_imag_avg,
    }


if __name__ == '__main__':
    BASE = '/home/nas/user/Uihyeop/NN_Zoo/TF_Restormer_release/models/TF_Restormer/inference_wav'

    datasets = [
        {
            'name': 'VCTK_DEMAND (denoising, 16kHz)',
            'target': f'{BASE}/VCTK_DEMAND_target_16kto16k',
            'output': f'{BASE}/VCTK_DEMAND_adversarial_to48k_baseline_16kto16k',
            'sr': 16000,
        },
        {
            'name': 'VCTK_SR_ND 8→16kHz (noisy-distorted SSR)',
            'target': f'{BASE}/VCTK_SR_ND_target_8kto16k',
            'output': f'{BASE}/VCTK_SR_ND_adversarial_to48k_baseline_8kto16k',
            'sr': 16000,
        },
        {
            'name': 'VCTK_SR_ND 8→44.1kHz (noisy-distorted SSR)',
            'target': f'{BASE}/VCTK_SR_ND_target_8kto44k',
            'output': f'{BASE}/VCTK_SR_ND_adversarial_to48k_baseline_8kto44k',
            'sr': 44100,
        },
        {
            'name': 'VCTK_SR_ND 16→48kHz (noisy-distorted SSR)',
            'target': f'{BASE}/VCTK_SR_ND_target_16kto48k',
            'output': f'{BASE}/VCTK_SR_ND_adversarial_to48k_baseline_16kto48k',
            'sr': 48000,
        },
        {
            'name': 'VCTK_SR 8→16kHz (clean SR)',
            'target': f'{BASE}/VCTK_SR_target_8kto16k',
            'output': f'{BASE}/VCTK_SR_adversarial_to48k_baseline_8kto16k',
            'sr': 16000,
        },
        {
            'name': 'VCTK_SR 8→44.1kHz (clean SR)',
            'target': f'{BASE}/VCTK_SR_target_8kto44k',
            'output': f'{BASE}/VCTK_SR_adversarial_to48k_baseline_8kto44k',
            'sr': 44100,
        },
        {
            'name': 'VCTK_SR 16→48kHz (clean SR)',
            'target': f'{BASE}/VCTK_SR_target_16kto48k',
            'output': f'{BASE}/VCTK_SR_adversarial_to48k_baseline_16kto48k',
            'sr': 48000,
        },
    ]

    all_results = []
    for ds in datasets:
        result = analyze_dataset(ds['target'], ds['output'], ds['sr'], ds['name'])
        if result:
            all_results.append(result)

    # Final cross-dataset summary
    print(f"\n{'=' * 70}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Per-Utterance Pearson r (target_RMS vs L1_mag):")
    for r in all_results:
        rp, _ = stats.pearsonr(r['utt_target_rms'], r['utt_l1_mag'])
        rs, _ = stats.spearmanr(r['utt_target_rms'], r['utt_l1_mag'])
        print(f"    {r['dataset']:<45} Pearson={rp:.4f}  Spearman={rs:.4f}")

    print(f"\n  Per-Freq-Bin Pearson r (E[|S(f)|] vs E[L1_mag(f)]):")
    for r in all_results:
        rp, _ = stats.pearsonr(r['freq_target_e_avg'], r['freq_l1_mag_avg'])
        rs, _ = stats.spearmanr(r['freq_target_e_avg'], r['freq_l1_mag_avg'])
        print(f"    {r['dataset']:<45} Pearson={rp:.4f}  Spearman={rs:.4f}")
