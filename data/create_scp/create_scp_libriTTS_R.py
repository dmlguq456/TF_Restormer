import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

train_mix_scp = 'data/scp/scp_LibriTTS_R/tr_s.scp'
# Get LibriTTS_R path from environment variable
libri_tts_r_root = os.getenv('LIBRI_TTS_R_DB_ROOT', '/data1/DB/LibriTTS_R')
train_mix = os.path.join(libri_tts_r_root, 'LibriTTS_R')


with open(train_mix_scp, 'w') as tr_mix:
    for root, dirs, files in os.walk(train_mix):
        if root == train_mix:
            dirs[:] = [d for d in dirs if d.startswith('train')]
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                tr_mix.write(f"{full_path} {full_path}\n")
                





dev_mix_scp = 'data/scp/scp_LibriTTS_R/cv_s.scp'

# Use the same path from environment variable
dev_mix = os.path.join(libri_tts_r_root, 'LibriTTS_R')



with open(dev_mix_scp, 'w') as cv_mix:
    for root, dirs, files in os.walk(dev_mix):
        if root == dev_mix:
            dirs[:] = [d for d in dirs if d.startswith('dev')]
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                cv_mix.write(f"{full_path} {full_path}\n")
                