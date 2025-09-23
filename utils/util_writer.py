import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')

import io
import PIL.Image
from torchvision.transforms import ToTensor
import librosa as rs


# https://pytorch.org/docs/stable/tensorboard.html

class MyWriter(SummaryWriter):
    def __init__(self, logdir, n_fft=512,n_hop=256,sr=16000):
        super(MyWriter, self).__init__(logdir,flush_secs=1)

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.sr = sr

        self.window = torch.hann_window(window_length=n_fft, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False)
    def log_value(self, train_loss, step,tag):
        self.add_scalar(tag, train_loss, step)

    def log_train(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_test(self,test_loss,step) : 
        self.add_scalar('test_loss', test_loss, step)

    def log_audio(self,wav,label='label',step=0) : 
        wav = wav.detach().cpu().numpy()
        wav = wav/np.max(np.abs(wav))
        self.add_audio(label, wav, step, self.sr)

    def log_MFCC(self,input,output,clean,step):
        input = input.to('cpu')
        output = output.to('cpu')
        clean= clean.to('cpu')

        noisy = input[0]
        estim = input[1]

        noisy = noisy.detach().numpy()
        estim = estim.detach().numpy()
        output = output.detach().numpy()
        clean= clean.detach().numpy()

        output = np.expand_dims(output,0)
        clean = np.expand_dims(clean,0)

        noisy = MFCC2plot(noisy)
        estim = MFCC2plot(estim)
        output = MFCC2plot(output)
        clean = MFCC2plot(clean)

        self.add_image('noisy',noisy,step,dataformats='HWC')
        self.add_image('estim',estim,step,dataformats='HWC')
        self.add_image('clean',clean,step,dataformats='HWC')
        self.add_image('output',output,step,dataformats='HWC')

        #self.add_image('noisy',noisy,step)
        #self.add_image('estim',estim,step)
        #self.add_image('output',output,step)

    # add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    def log_spec(self,data,label,step) :
        self.add_image(label,
            spec2plot(data), step, dataformats='HWC')

    def log_mag(self,data,label,step):
        self.add_image(label,
            mag2plot(data), step, dataformats='HWC')
 
    def log_wav2spec(self,src,key,step) :
        # src = src/torch.max(torch.abs(src))
        src = torch.stft(src,n_fft=self.n_fft, hop_length = self.n_hop, window = self.window.to(src.device), center = True, normalized=False, onesided=True, return_complex=True)

        self.log_spec(src, key, step)

    
    """
    data : 
        (9, n_sample) == [noisy, target 0 ~ target 3, output 0 ~ output 3]
    """
    def log_DOA_wav(self,data,step,label="Output"):
        image = wav2plotDOA(data)
        self.add_image(label,image,step)



def fig2np(fig):
    # 캔버스를 그려서 최신 버퍼 내용 확보
    fig.canvas.draw()

    # 너비, 높이 가져오기
    w, h = fig.canvas.get_width_height()

    # RGBA 버퍼 → numpy 배열
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape((h, w, 4))[..., :3]  # RGB만 추출

    return data


# def fig2np(fig):
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     return data

def MFCC2plot(MFCC):
    MFCC = np.transpose(MFCC)
    fig, ax = plt.subplots()
    im = plt.imshow(MFCC, cmap=cm.jet, aspect='auto')
    plt.colorbar(im)
    plt.clim(-80,20)
    fig.canvas.draw()

    data = fig2np(fig)
    plt.close()
    return data

def spec2plot(data,normalized=True):
    data = data.detach().cpu().numpy()
    n_shape = len(data.shape)
    mag = np.abs(data)

    np.seterr(divide = 'warn') 

    mag = 10*np.log(mag)
    fig, ax = plt.subplots()
    im = plt.imshow(mag, cmap=cm.jet, aspect='auto',origin='lower')
    plt.colorbar(im)
    plt.clim(-80,20)
    
    plt.xlabel('Time')
    plt.ylabel('Freq')
    
    fig.canvas.draw()
    plot = fig2np(fig)
    return plot

def mag2plot(data):
    mag = data.detach().cpu().numpy()
    mag = 10*np.log(mag)
    fig, ax = plt.subplots()
    im = plt.imshow(mag, cmap=cm.jet, aspect='auto',origin='lower')
    plt.colorbar(im)
    plt.clim(-80,20)
    
    plt.xlabel('Time')
    plt.ylabel('Freq')
    
    fig.canvas.draw()
    plot = fig2np(fig)
    return plot

"""
wav2plotDOA : 
    wavfrom 2 tensorboard image
    Impelemented for DOA-Separation

input
    + waveform : numpy array[9,n_sample]
"""
def wav2plotDOA(waveform, sample_rate=16000):
    num_channels = waveform.shape[0]
    num_frames = waveform.shape[1]
    time_axis = np.arange(start=0, stop=num_frames) / sample_rate

    figure, axes = plt.subplots(2, 4)

    ## input plotting routine 
    #gs = axes[0,0].get_gridspec()
    #for ax in axes[0,:]:
    #    ax.remove()
    # big = figure.add_subplot(gs[0,:])
    # big.set_title('input')
    # big.plot(time_axis, waveform[0], linewidth=1)
        
    
    for c in range(4):
        idx_y = 0
        idx_x = c
        
        axes[idx_y,idx_x].plot(time_axis, waveform[0+c], linewidth=1)
        axes[idx_y,idx_x].grid(True)
        axes[idx_y,idx_x].set_title(f'target {c}')
        
    for c in range(4):
        idx_y = 1
        idx_x = c
        
        axes[idx_y,idx_x].plot(time_axis, waveform[4+c], linewidth=1)
        axes[idx_y,idx_x].grid(True)
        axes[idx_y,idx_x].set_title(f'output {c}')
        
    figure.set_size_inches(14, 10)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image




if __name__=='__main__':
    MyWriter()

