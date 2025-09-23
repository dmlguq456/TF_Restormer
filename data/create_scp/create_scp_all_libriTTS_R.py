#!/usr/bin/env python3
"""
Create all SCP files for LibriTTS_R dataset including noise files.
This script uses environment variables from .env file.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get paths from environment variables
libri_tts_r_root = os.getenv('LIBRI_TTS_R_DB_ROOT', '/data1/DB/LibriTTS_R')
dns_db_root = os.getenv('DNS_DB_ROOT', '/data1/DB/DNS-Challenge-16kHz')
dns_noise_subpath = os.getenv('DNS_NOISE_PATH', 'datasets_fullband/noise_fullband')

# Paths - LibriTTS_R data is directly in the root
train_mix = libri_tts_r_root
dev_mix = libri_tts_r_root
dns_noise_path = os.path.join(dns_db_root, dns_noise_subpath)

# Output SCP files
train_spk_scp = 'data/scp/scp_LibriTTS_R/tr_s.scp'
dev_spk_scp = 'data/scp/scp_LibriTTS_R/cv_s.scp'
train_noise_scp = 'data/scp/scp_LibriTTS_R/tr_n.scp'

# Create output directory if it doesn't exist
os.makedirs('data/scp/scp_LibriTTS_R', exist_ok=True)

print("Creating SCP files for LibriTTS_R dataset...")
print(f"LibriTTS_R root: {libri_tts_r_root}")
print(f"DNS noise path: {dns_noise_path}")

# Create train speaker SCP
print(f"\nCreating train speaker SCP: {train_spk_scp}")
with open(train_spk_scp, 'w') as tr_spk:
    count = 0
    for root, dirs, files in os.walk(train_mix):
        if root == train_mix:
            dirs[:] = [d for d in dirs if d.startswith('train')]
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                tr_spk.write(f"{full_path} {full_path}\n")
                count += 1
    print(f"  Added {count} train speaker files")

# Create dev speaker SCP
print(f"\nCreating dev speaker SCP: {dev_spk_scp}")
with open(dev_spk_scp, 'w') as cv_spk:
    count = 0
    for root, dirs, files in os.walk(dev_mix):
        if root == dev_mix:
            dirs[:] = [d for d in dirs if d.startswith('dev')]
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                cv_spk.write(f"{full_path} {full_path}\n")
                count += 1
    print(f"  Added {count} dev speaker files")

# Create noise SCP
print(f"\nCreating noise SCP: {train_noise_scp}")
if os.path.exists(dns_noise_path):
    with open(train_noise_scp, 'w') as tr_n:
        count = 0
        for root, dirs, files in os.walk(dns_noise_path):
            files.sort()
            for file in files:
                if file.lower().endswith('.wav'):
                    full_path = os.path.join(root, file)
                    # SCP format: filename full_path
                    tr_n.write(f"{file} {full_path}\n")
                    count += 1
        print(f"  Added {count} noise files")
else:
    print(f"  Warning: DNS noise path does not exist: {dns_noise_path}")
    print("  Creating empty noise scp file")
    with open(train_noise_scp, 'w') as tr_n:
        tr_n.write("# Empty noise scp file - DNS noise data not found\n")

print("\nSCP file creation completed!")
print(f"Files created in: data/scp/scp_LibriTTS_R/")