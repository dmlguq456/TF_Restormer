#!/usr/bin/env python3
"""
Quick NISQA test to verify parameters
"""
import numpy as np
from util_nisqa import NISQAModel

# Create dummy audio (3 seconds at 48kHz)
fs = 48000
duration = 3.0
t = np.linspace(0, duration, int(fs * duration))
# Pure tone at 440Hz with some noise
audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

# Initialize NISQA
print("Initializing NISQA...")
nisqa = NISQAModel()

print(f"\n=== NISQA Parameters ===")
for k, v in nisqa.args.items():
    print(f"{k}: {v}")

# Calculate actual values for 48kHz
print(f"\n=== Calculated for {fs}Hz ===")
hop_samples = int(fs * nisqa.args['ms_hop_length'])
win_samples = int(fs * nisqa.args['ms_win_length'])
print(f"hop_length: {hop_samples} samples ({nisqa.args['ms_hop_length']*1000:.1f}ms)")
print(f"win_length: {win_samples} samples ({nisqa.args['ms_win_length']*1000:.1f}ms)")
print(f"n_fft: {nisqa.args['ms_n_fft']} samples")
print(f"Zero padding: {nisqa.args['ms_n_fft'] - win_samples} samples")

# Predict MOS
print(f"\n=== Prediction ===")
print(f"Audio: {audio.shape}, fs={fs}Hz, duration={duration}s")
mos = nisqa.forward(audio, fs)
print(f"MOS: {mos:.3f}")

# Test with different sample rate
print(f"\n=== Test with 16kHz ===")
audio_16k = audio[::3]  # Simple downsampling
fs_16k = 16000
hop_samples_16k = int(fs_16k * nisqa.args['ms_hop_length'])
win_samples_16k = int(fs_16k * nisqa.args['ms_win_length'])
print(f"hop_length: {hop_samples_16k} samples ({nisqa.args['ms_hop_length']*1000:.1f}ms)")
print(f"win_length: {win_samples_16k} samples ({nisqa.args['ms_win_length']*1000:.1f}ms)")
print(f"n_fft: {nisqa.args['ms_n_fft']} samples")
print(f"Zero padding: {nisqa.args['ms_n_fft'] - win_samples_16k} samples")
mos_16k = nisqa.forward(audio_16k, fs_16k)
print(f"MOS: {mos_16k:.3f}")
