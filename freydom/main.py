import audiofile
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from sklearn.preprocessing import minmax_scale
from scipy.fft import rfft, irfft, rfftfreq

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

        filename = f'{basename(self.filename)}-processed.{splitext(self.filename)[1]}'
        audiofile.write(filename, (1, waveform), self.sampling_rate, self.bit_depth, True)


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

    block_bins = None
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
        print(f'Block size:      {block_size}')
        print(f'Frame size:      {self.frame_size}s')

        self.y_min = 0.0001
        self.y_max = 90.0
        self.x_min = 0.0
        self.x_max = 22000.0

        print(f'Y-min:           {self.y_min}dB')
        print(f'Y-max:           {self.y_max}dB')
        print(f'X-min:           {self.x_min}Hz')
        print(f'X-max:           {self.x_max}Hz')

        block_indices = np.arange(0, self.n_samples-1, block_size)

        self.block_bins = np.ndarray(shape=(1, block_indices.size), dtype=list)
        self.block_bins_meta = []

        print('\nbuilding rfft array..')
        for block_index in block_indices:
            try:
                self.process_block(block_index)
            except ValueError as e:
                print(f'error: {e}')

        print(repr(self.block_bins))
        exit()

        waveform = np.ndarray(shape=self.waveform.shape)

        # TODO: - calculate temporal variance for each bin in block_bins
        #       - select 3 bands with highest variance and average them at each block_bins (axis 0),
        #         store the resulting float value in block_bins_roi
        #       - interpolate block_bins_roi to match self.n_samples size
        #           - block_bins_roi.shape=(22000, block_bins[0].size)
        #           - scipy.fft.irfft(block_bins_roi[normalized 0.0-1.0] * fft_data, n=n_samples)
        #       - convolve self.waveform with block_bins_roi in frequency domain and store in waveform
        #           - scipy.signal.fftconvolve
        #       - save waveform
        self.save(np.ndarray(waveform))

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

        fft_orig_min = min(fft_data)     # type: float
        fft_orig_max = max(fft_data)     # type: float

        fft_pwr = (np.abs(fft_data) / float(self.block_size)) ** 2  # type: np.ndarray
        fft_pwr = minmax_scale(fft_pwr, (self.y_min, self.y_max))
        fft_pwr = np.nan_to_num(fft_pwr, nan=self.y_min, posinf=self.y_max, neginf=self.y_min)
        fft_pwr = 10 * np.log10(fft_pwr)

        fft_out = np.ndarray(shape=(3,), dtype=list)
        fft_out[0] = fft_pwr.reshape(fft_pwr.size, 1).tolist()
        fft_out[1] = [fft_orig_min, fft_orig_max]

        np.append(self.block_bins, fft_out)

        # x = rfftfreq(n=self.block_size - 1, d=self.sampling_period)

        # axes = plt.axes()
        # axes.set_xbound(self.x_min, self.x_max)
        # axes.set_xlim(self.x_min, self.x_max)
        # axes.set_ybound(self.y_min, self.y_max)
        # axes.set_ylim(self.y_min, self.y_max)
        # axes.plot(x, y)
        #
        # plt.axes(axes)
        # plt.show()
