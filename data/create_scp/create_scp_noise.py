import os



train_noise_scp = 'tr_n.scp'

cv_noise_scp = 'cv_n.scp'

test_noise_scp = 'tt_n.scp'

train_noise = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/colorednoise_v9_short/tr'

test_noise = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/colorednoise_v9_short/tt'

cv_noise = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/colorednoise_v9_short/cv'


tr_s2 = open(train_noise_scp,'w')
for root, dirs, files in os.walk(train_noise):
    files.sort()
    for file in files:
        tr_s2.write(file+" "+root+'/'+file)
        tr_s2.write('\n')


tt_s2 = open(test_noise,'w')
for root, dirs, files in os.walk(test_noise):
    files.sort()
    for file in files:
        tt_s2.write(file+" "+root+'/'+file)
        tt_s2.write('\n')


cv_s2_file = open(cv_noise_scp,'w')
for root, dirs, files in os.walk(cv_noise):
    files.sort()
    for file in files:
        cv_s2_file.write(file+" "+root+'/'+file)
        cv_s2_file.write('\n')