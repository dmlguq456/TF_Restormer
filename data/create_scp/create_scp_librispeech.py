import os



# train_mix_scp = 'data/scp/scp_LS/tr_s_100.scp'

# train_mix = '/home/nas4/DB/LibriSpeech/librispeech_wav_convert/train-clean-100'

# tr_mix = open(train_mix_scp,'w')
# for root, dirs, files in os.walk(train_mix):
#     files.sort()
#     for file in files:
#         tr_mix.write(file+" "+root+'/'+file)
#         tr_mix.write('\n')


train_noise_scp = 'data/scp/scp_LS/tr_n.scp'

train_noise = '/home/nas/user/Uihyeop/DB/Reverb_challenge/reverb_tools_for_Generate_mcTrainData/NOISE_mono'

tr_noise = open(train_noise_scp,'w')
for root, dirs, files in os.walk(train_noise):
    files.sort()
    for file in files:
        tr_noise.write(file+" "+root+'/'+file)
        tr_noise.write('\n')



# dev_mix_scp = 'scp_DNS/cv_s.scp'

# dev_mix = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/min/cv/s1'

# cv_mix = open(dev_mix_scp,'w')
# for root, dirs, files in os.walk(dev_mix):
#     files.sort()
#     for file in files:
#         cv_mix.write(file+" "+root+'/'+file)
#         cv_mix.write('\n')