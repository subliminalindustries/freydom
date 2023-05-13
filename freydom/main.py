
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from sklearn.preprocessing import minmax_scale
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import normalize

from .utils import AudioFile


class FreyVoiceEnhancer:

    def __init__(self, filename):
        super(FreyVoiceEnhancer, self).__init__()

        self.file = AudioFile(filename)

    def process(self):

        waveform = self.file.signal[0]      # type: np.ndarray
        n_samples = waveform.size

        Fs = self.file.sampling_rate        # Sampling rate = Hz
        Ts = 1.0 / self.file.sampling_rate  # Sampling period = time (s) to acquire 1 sample
        N = 128                             # Block size = number of samples for fourier transform
        T = N * Ts                          # Frame size = time (s) to sample 1 block -> N * Ts or N/Fs
        BW = Fs / 2.0                       # Bandwidth
        SL = round(N / 2 + 1)               # Spectral lines (no. of frequency samples)
        f = np.linspace(0, BW, SL)          # Frequency scale

        print(f'Total samples:   {n_samples}')
        print(f'Sample rate:     {Fs}Hz')
        print(f'Sampling period: {Ts}s')
        print(f'Block size:      {N}')
        print(f'Frame size:      {T}s')

        y_min = 0.0
        y_max = 90.0
        x_min = 0.0
        x_max = 22000.0

        print(f'Y-min: {y_min}')
        print(f'Y-max: {y_max}')
        print(f'X-min: {x_min}')
        print(f'X-max: {x_max}')

        last = None
        block_idx = np.arange(0, n_samples-1, N)
        for idx in block_idx:
            try:
                if idx + N > n_samples:
                    last = (idx + N) - n_samples - 1

                if last is not None:
                    chunk = waveform[idx:n_samples-1]
                else:
                    chunk = waveform[idx:idx+N-1]

                fft_data = rfft(chunk)  # type: np.ndarray
                pwr_data = (np.abs(fft_data) / float(N)) ** 2

                y = minmax_scale(10 * np.log10(pwr_data), (y_min, y_max))
                x = rfftfreq(n=N-1, d=Ts)

                axes = plt.axes()   # type: Axes
                axes.set_xbound(x_min, x_max)
                axes.set_xlim(x_min, x_max)
                axes.set_ybound(y_min, y_max)
                axes.set_ylim(y_min, y_max)
                axes.plot(x, y)

                plt.axes(axes)
                plt.show()

            except ValueError as e:
                print(f'error: {e}')
                continue
