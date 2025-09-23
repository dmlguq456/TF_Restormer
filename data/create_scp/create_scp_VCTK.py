import os



train_mix_scp = 'data/scp/scp_VCTK/tr_s.scp'
train_mix = "/home/DB/VCTK/wav48/tr/"


with open(train_mix_scp, 'w') as tr_mix:
    for root, dirs, files in os.walk(train_mix):
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                tr_mix.write(f"{full_path} {full_path}\n")
                




dev_mix_scp = 'data/scp/scp_VCTK/cv_s.scp'

dev_mix = "/home/DB/VCTK/wav48/cv/"



with open(dev_mix_scp, 'w') as cv_mix:
    for root, dirs, files in os.walk(dev_mix):
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                cv_mix.write(f"{full_path} {full_path}\n")
                