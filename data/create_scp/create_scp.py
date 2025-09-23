import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Option to use relative paths (can be configured)
USE_RELATIVE_PATHS = True

def format_path(filepath, use_relative=USE_RELATIVE_PATHS):
    """Format path based on relative/absolute preference"""
    if use_relative:
        try:
            return os.path.relpath(filepath, os.getcwd())
        except ValueError:
            return filepath
    return filepath

# Get paths from environment variables
scp_root = os.path.join(os.getenv('SCP_ROOT', 'data/scp'), 'scp_DNS')
os.makedirs(scp_root, exist_ok=True)

# DNS Dataset - Clean
dns_db_root = os.getenv('DNS_DB_ROOT', '/home/nas/user/Uihyeop/DB/DNS-Challenge-16kHz')
dns_clean_path = os.getenv('DNS_CLEAN_PATH', 'datasets/clean')
train_mix = os.path.join(dns_db_root, dns_clean_path)

train_mix_scp = os.path.join(scp_root, 'tr_s.scp')
with open(train_mix_scp, 'w') as tr_mix:
    for root, dirs, files in os.walk(train_mix):
        files.sort()
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                formatted_path = format_path(full_path)
                tr_mix.write(f"{file} {formatted_path}\n")

# DNS Dataset - Noise
dns_noise_path = os.getenv('DNS_NOISE_PATH', 'datasets/noise')
train_noise = os.path.join(dns_db_root, dns_noise_path)

train_noise_scp = os.path.join(scp_root, 'tr_n.scp')
with open(train_noise_scp, 'w') as tr_noise:
    for root, dirs, files in os.walk(train_noise):
        files.sort()
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                formatted_path = format_path(full_path)
                tr_noise.write(f"{file} {formatted_path}\n")

# WSJ0 Dataset - CV
wsj0_db_root = os.getenv('WSJ0_DB_ROOT', '/home/nas/user/Uihyeop/DB/wsj0-mix')
wsj0_cv_path = os.getenv('WSJ0_CV_PATH', '2speakers/wav16k/min/cv/s1')
dev_mix = os.path.join(wsj0_db_root, wsj0_cv_path)

dev_mix_scp = os.path.join(scp_root, 'cv_s.scp')
with open(dev_mix_scp, 'w') as cv_mix:
    for root, dirs, files in os.walk(dev_mix):
        files.sort()
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                formatted_path = format_path(full_path)
                cv_mix.write(f"{file} {formatted_path}\n")