import os



# train_mix_scp = 'data/scp/scp_AI_HUB_kspon/tr_AKS.scp'
# train_mix_scp = 'data/scp/scp_AI_HUB_kspon/tr_AKS.scp'
# train_mix = "/home/DB/KsponSpeech_wav"

# tr_mix = open(train_mix_scp,'w')
# for root, dirs, files in os.walk(train_mix):
#     files.sort()
#     for file in files:
#         tr_mix.write(file+" "+root+'/'+file)
#         tr_mix.write('\n')

train_mix_scp = 'data/scp/scp_OLKAVS_48k/tr_OLKAVS.scp'
train_mix = "/home/DB/OLKAVS_wav"

# tr_mix = open(train_mix_scp,'w')
# for root, dirs, files in os.walk(train_mix):
#     files.sort()
#     for file in files:
#         tr_mix.write(file+" "+root+'/'+file)~
#         tr_mix.write('\n')

with open(train_mix_scp, 'w') as tr_mix:
    for root, dirs, files in os.walk(train_mix):
        files.sort()
        for file in files:
            full_path = os.path.join(root, file)
            tr_mix.write(f"{full_path} {full_path}\n")

# train_noise_scp = 'scp_DNS/tr_n.scp'

# train_noise = '/home/nas/user/Uihyeop/DB/DNS-Challenge-16kHz/datasets/noise'

# tr_noise = open(train_noise_scp,'w')
# for root, dirs, files in os.walk(train_noise):
#     files.sort()
#     for file in files:
#         tr_noise.write(file+" "+root+'/'+file)
#         tr_noise.write('\n')



# dev_mix_scp = 'scp_DNS/cv_s.scp'

# dev_mix = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/min/cv/s1'

# cv_mix = open(dev_mix_scp,'w')
# for root, dirs, files in os.walk(dev_mix):
#     files.sort()
#     for file in files:
#         cv_mix.write(file+" "+root+'/'+file)
#         cv_mix.write('\n')