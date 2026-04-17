import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')


# https://pytorch.org/docs/stable/tensorboard.html

class TBWriter(SummaryWriter):
    def __init__(self, logdir: str, n_fft: int = 512, n_hop: int = 256, sr: int = 16000) -> None:
        super(TBWriter, self).__init__(logdir,flush_secs=1)

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.sr = sr

        self.window = torch.hann_window(window_length=n_fft, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False)
    def log_audio(self, wav: torch.Tensor, label: str = 'label', step: int = 0, normalize: bool = True) -> None:
        wav = wav.detach().cpu().numpy()
        if normalize:
            wav = wav / (np.max(np.abs(wav)) + 1.0e-3)
        self.add_audio(label, wav, step, self.sr)

    # add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    def log_spec(self, data: torch.Tensor, label: str, step: int) -> None:
        self.add_image(label,
            spec_to_plot(data), step, dataformats='HWC')

    def log_mag(self,data,label,step):
        self.add_image(label,
            mag_to_plot(data), step, dataformats='HWC')

    def log_wav2spec(self, src: torch.Tensor, key: str, step: int, normalize: bool = False) -> None:
        if normalize:
            src = src / (torch.max(torch.abs(src)) + 1.0e-3)
        src = torch.stft(src, n_fft=self.n_fft, hop_length=self.n_hop,
                         window=self.window.to(src.device), center=True,
                         normalized=False, onesided=True, return_complex=True)

        self.log_spec(src, key, step)


def fig_to_np(fig):
    """Convert a matplotlib figure to a numpy RGB array."""
    w, h = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape((h, w, 4))[..., :3]  # RGB only
    return data


def _array_to_plot(data, origin='lower', clim=(-80, 20), xlabel=None, ylabel=None):
    """Convert a 2D array to a numpy image array via matplotlib."""
    fig, ax = plt.subplots()
    im = plt.imshow(data, cmap=cm.jet, aspect='auto', origin=origin)
    plt.colorbar(im)
    plt.clim(*clim)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    fig.canvas.draw()
    img = fig_to_np(fig)
    plt.close(fig)
    return img


def spec_to_plot(data, normalized=True):
    np.seterr(divide='warn')
    data = data.detach().cpu().numpy()
    mag = np.abs(data)
    mag = 10 * np.log(mag)
    return _array_to_plot(mag, xlabel='Time', ylabel='Freq')

def mag_to_plot(data):
    mag = data.detach().cpu().numpy()
    mag = 10 * np.log(mag)
    return _array_to_plot(mag, xlabel='Time', ylabel='Freq')
