import audiofile
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
from scipy.interpolate import CubicSpline
from scipy import signal
from os.path import realpath, basename, splitext


class AudioFile(object):
    """
    Audio File Interface

    Loads audio files and allows saving modified waveform.

    Attributes
    ----------
    band_width: int
        sampling rate / 2
    bit_depth: int
        bit depth
    duration: float
        duration in seconds
    filename: str
        filename
    n_samples: int
        total number of samples
    sampling_period: int
        time in seconds needed to acquire one block of samples
    sampling_rate: int
        number of samples per second
    waveform: np.ndarray
        raw audio data
    """

    band_width = None
    bit_depth = None
    duration = None
    filename = None
    n_samples = None
    sampling_period = None
    sampling_rate = None
    waveform = None

    def __init__(self):
        """
        Constructor
        """

        super(AudioFile, self).__init__()

    def load(self, filename: str):
        """
        Loads an audio file

        :param filename: str
        """
        self.filename = realpath(filename)

        signal, self.sampling_rate = audiofile.read(filename, always_2d=True)
        self.waveform = signal[0]

        self.bit_depth = audiofile.bit_depth(filename)
        self.duration = audiofile.duration(filename)

        self.band_width = self.sampling_rate / 2.0
        self.n_samples = self.waveform.size
        self.sampling_period = 1.0 / self.sampling_rate

    def save(self, waveform: np.ndarray):
        """
        Saves waveform to filename with added suffix '-processed':

            input.wav becomes input-processed.wav

        :param waveform: np.ndarray
        """

        ext = splitext(self.filename)[1]
        filename = basename(self.filename).replace(ext, '')
        filename = f'./data/{filename}-processed{ext}'
        audiofile.write(filename, waveform, self.sampling_rate, self.bit_depth, True)
        print(f'wrote {filename}')


class FreyVoiceEnhancer(AudioFile):
    """
    Frey Voice Enhancer

    Increases the signal-to-noise ratio for body-proximal recordings of the microwave auditory effect.

    Attributes
    ----------
    block_size: int
        number of samples to acquire for each fft transform
    frame_size: int
        time (s) to sample 1 block:
        (block_size * file sampling_period) or (block_size / file sampling_rate)
    freq_scale: list
        frequency scale
    spectral_lines: int
        number of frequency samples
    x_max: float
        chart maximum x value
    x_min: float
        chart minimum x value
    y_max: float
        chart maximum y value
    y_min: float
        chart minimum y value
    """

    file = None
    filename = None

    output_data = None
    block_bins_meta = None
    block_size = None

    frame_size = None
    freq_scale = None
    spectral_lines = None

    x_max = None
    x_min = None
    y_max = None
    y_min = None

    def __init__(self):
        super(FreyVoiceEnhancer, self).__init__()

    def process(self, filename: str, block_size: int = 128):
        """
        Process an audio file

        Parameters
        ________
        :param filename: str
            path to audio file
        :param block_size: int
            number of samples to acquire for each fft transform
        """

        self.load(filename)

        self.block_size = block_size
        self.frame_size = block_size * self.sampling_period
        self.spectral_lines = int(round(block_size / 2 + 1))
        self.freq_scale = np.linspace(0, self.band_width, self.spectral_lines)

        print(f'Filename:        {self.filename}\n')
        print(f'Total samples:   {self.n_samples}')
        print(f'Sample rate:     {self.sampling_rate}Hz')
        print(f'Sampling period: {self.sampling_period}s')
        print(f'Block size:      {self.block_size}')
        print(f'Frame size:      {self.frame_size}s')

        self.y_min = 0.0000001
        self.y_max = 1.0
        self.x_min = 0.0
        self.x_max = 22000.0

        print(f'Y-min:           {self.y_min}')
        print(f'Y-max:           {self.y_max}')
        print(f'X-min:           {self.x_min}Hz')
        print(f'X-max:           {self.x_max}Hz')

        block_indices = np.arange(0, self.n_samples-1, self.block_size)

        self.output_data = []

        print('\nFiltering..')
        for block_index in block_indices:
            try:
                self.process_block(block_index)
            except ValueError as e:
                print(f'error: {e}')

        wav_fft = rfft(self.waveform)

        x = np.arange(0, self.n_samples, self.block_size).astype(dtype=list)
        mult_sig_x = np.arange(0, (self.n_samples // 2)+1).astype(dtype=list)
        interp = CubicSpline(x, self.output_data)
        mult_sig_y = interp(mult_sig_x)

        # multiply signal by filter and subtract filter
        out_fft = np.subtract(np.multiply(wav_fft, mult_sig_y), mult_sig_y)
        res = irfft(out_fft)

        # axes = plt.axes()
        # axes.set_ybound(-1, 1)
        # axes.set_ylim(-1, 1)
        # axes.plot(np.arange(0, self.waveform.size), self.waveform)
        # axes.plot(np.arange(0, res.size), res)
        # plt.axes(axes)
        # plt.show()

        # exit()
        self.save(res)

    def process_block(self, block_index: int):
        """
        Process block at block_index

        Parameters
        ________
        :param block_index:
            Index of next block in waveform to process
        """
        last = None
        if block_index + self.block_size > self.n_samples:
            last = (block_index + self.block_size) - self.n_samples - 1

        if last is not None:
            chunk = self.waveform[block_index:self.n_samples - 1]
        else:
            chunk = self.waveform[block_index:block_index + self.block_size - 1]

        fft_data = rfft(chunk)           # type: np.ndarray

        fft_lin = np.abs(fft_data)
        fft_lin = minmax_scale(fft_lin, (self.y_min, 1.0))
        fft_lin = np.nan_to_num(fft_lin, nan=self.y_min, posinf=1.0, neginf=self.y_min)

        # x = rfftfreq(n=self.block_size - 1, d=self.sampling_period)
        y = fft_lin

        # The 0th bin contains the most information about the changes in amplitude due to v2k speech
        self.output_data.append(y[0])

        # axes = plt.axes()
        # axes.set_xbound(0, self.x_max)
        # axes.set_xlim(0, self.x_max)
        # axes.set_ybound(self.y_min, self.y_max)
        # axes.set_ylim(self.y_min, self.y_max)
        # axes.hist2d(x, y, self.block_size)
        # plt.axes(axes)
        # plt.show()
