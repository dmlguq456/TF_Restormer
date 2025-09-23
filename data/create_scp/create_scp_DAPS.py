import os



train_mix_scp = 'data/scp/scp_DAPS/tr_s.scp'
train_mix = "/home/DB/DAPS/clean_segmented_2"


with open(train_mix_scp, 'w') as tr_mix:
    for root, dirs, files in os.walk(train_mix):
        if root == train_mix:
            dirs[:] = [d for d in dirs if d.startswith('train')]
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                tr_mix.write(f"{full_path} {full_path}\n")
                




dev_mix_scp = 'data/scp/scp_DAPS/cv_s.scp'

dev_mix = "/home/DB/DAPS/clean_segmented_2"



with open(dev_mix_scp, 'w') as cv_mix:
    for root, dirs, files in os.walk(dev_mix):
        if root == dev_mix:
            dirs[:] = [d for d in dirs if d.startswith('dev')]
        files.sort()
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                cv_mix.write(f"{full_path} {full_path}\n")
                