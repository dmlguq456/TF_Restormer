import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Get paths from environment variables
db_root = os.getenv('WHAM_DB_ROOT', '/home/work/data/WHAM')
scp_root = os.path.join(os.getenv('SCP_ROOT', 'data/scp'), 'scp_ss_8k_wham')
os.makedirs(scp_root, exist_ok=True)

# SCP file paths (relative to project root)
train_mix_scp = os.path.join(scp_root, 'tr_mix.scp')
train_s1_scp = os.path.join(scp_root, 'tr_s1.scp')
train_s2_scp = os.path.join(scp_root, 'tr_s2.scp')
train_n_scp = os.path.join(scp_root, 'tr_n.scp')

# Dataset paths (absolute)
train_mix = os.path.join(db_root, 'tr/mix_both')
train_s1 = os.path.join(db_root, 'tr/s1')
train_s2 = os.path.join(db_root, 'tr/s2')
train_n = os.path.join(db_root, 'tr/noise')


# Write train mix SCP with relative paths from DB_ROOT
with open(train_mix_scp, 'w') as tr_mix:
    for root, dirs, files in os.walk(train_mix):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            # Store relative path from DB_ROOT in SCP file
            rel_path = os.path.relpath(full_path, db_root)
            tr_mix.write(f"{file} {rel_path}\n")


# Write train s1 SCP with relative paths from DB_ROOT
with open(train_s1_scp, 'w') as tr_s1:
    for root, dirs, files in os.walk(train_s1):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            tr_s1.write(f"{file} {rel_path}\n")


# Write train s2 SCP with relative paths from DB_ROOT
with open(train_s2_scp, 'w') as tr_s2:
    for root, dirs, files in os.walk(train_s2):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            tr_s2.write(f"{file} {rel_path}\n")



# Write train noise SCP with relative paths from DB_ROOT
with open(train_n_scp, 'w') as tr_noise:
    for root, dirs, files in os.walk(train_n):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            tr_noise.write(f"{file} {rel_path}\n")




# Test SCP file paths
test_mix_scp = os.path.join(scp_root, 'tt_mix.scp')
test_s1_scp = os.path.join(scp_root, 'tt_s1.scp')
test_s2_scp = os.path.join(scp_root, 'tt_s2.scp')

# Test dataset paths
test_mix = os.path.join(db_root, 'tt/mix_both')
test_s1 = os.path.join(db_root, 'tt/s1')
test_s2 = os.path.join(db_root, 'tt/s2')

# Write test mix SCP with relative paths from DB_ROOT
with open(test_mix_scp, 'w') as tt_mix:
    for root, dirs, files in os.walk(test_mix):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            tt_mix.write(f"{file} {rel_path}\n")


# Write test s1 SCP with relative paths from DB_ROOT
with open(test_s1_scp, 'w') as tt_s1:
    for root, dirs, files in os.walk(test_s1):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            tt_s1.write(f"{file} {rel_path}\n")


# Write test s2 SCP with relative paths from DB_ROOT
with open(test_s2_scp, 'w') as tt_s2:
    for root, dirs, files in os.walk(test_s2):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            tt_s2.write(f"{file} {rel_path}\n")

# CV SCP file paths
cv_mix_scp = os.path.join(scp_root, 'cv_mix.scp')
cv_s1_scp = os.path.join(scp_root, 'cv_s1.scp')
cv_s2_scp = os.path.join(scp_root, 'cv_s2.scp')

# CV dataset paths
cv_mix = os.path.join(db_root, 'cv/mix_both')
cv_s1 = os.path.join(db_root, 'cv/s1')
cv_s2 = os.path.join(db_root, 'cv/s2')

# Write CV mix SCP with relative paths from DB_ROOT
with open(cv_mix_scp, 'w') as cv_mix_file:
    for root, dirs, files in os.walk(cv_mix):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            cv_mix_file.write(f"{file} {rel_path}\n")


# Write CV s1 SCP with relative paths from DB_ROOT
with open(cv_s1_scp, 'w') as cv_s1_file:
    for root, dirs, files in os.walk(cv_s1):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            cv_s1_file.write(f"{file} {rel_path}\n")


# Write CV s2 SCP with relative paths from DB_ROOT
with open(cv_s2_scp, 'w') as cv_s2_file:
    for root, dirs, files in os.walk(cv_s2):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, db_root)
            cv_s2_file.write(f"{file} {rel_path}\n")