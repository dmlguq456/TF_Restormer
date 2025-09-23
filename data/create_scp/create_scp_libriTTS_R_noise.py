import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Output scp file path
train_noise_scp = 'data/scp/scp_LibriTTS_R/tr_n.scp'

# DNS Challenge noise directory from environment variables
dns_db_root = os.getenv('DNS_DB_ROOT', '/data1/DB/DNS-Challenge-16kHz')
dns_noise_subpath = os.getenv('DNS_NOISE_PATH', 'datasets_fullband/noise_fullband')
dns_noise_path = os.path.join(dns_db_root, dns_noise_subpath)

# Create noise scp file
with open(train_noise_scp, 'w') as tr_n:
    if os.path.exists(dns_noise_path):
        for root, dirs, files in os.walk(dns_noise_path):
            files.sort()
            for file in files:
                if file.lower().endswith('.wav'):
                    full_path = os.path.join(root, file)
                    # SCP format: filename full_path
                    tr_n.write(f"{file} {full_path}\n")
        print(f"Created noise scp file: {train_noise_scp}")
    else:
        print(f"Warning: DNS noise path does not exist: {dns_noise_path}")
        print("Creating empty noise scp file for now")
        # You may need to update the path or download the DNS dataset