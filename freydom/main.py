import numpy as np

from sklearn.preprocessing import minmax_scale
from scipy.fft import rfft, irfft, rfftfreq

from .waveform import WaveForm


class FreyVoiceEnhancer(WaveForm):
    """
    Frey Voice Enhancer

    Increases the signal-to-noise ratio for body-proximal recordings of the microwave auditory effect.

    Attributes
    ----------
    :ivar flt_spectral: list
        spectral filter
    :ivar block_size: int
        number of samples to acquire for each fft transform
    :ivar convolve: bool
        whether to convolve the input waveform with the filter output
    :ivar filename: str
        path to audio file
    :ivar flt_band_start: float
        start frequency for temporal filter
    :ivar flt_band_stop: float
        stop frequency for temporal filter
    :ivar flt_bands: list
        contains lists containing filter band lower and upper cutoff frequencies
    :ivar frame_size: int
        time (s) to sample 1 block:
        (block_size * file sampling_period) or (block_size / file sampling_rate)
    :ivar freq_scale: list
        frequency scale
    :ivar spectral_lines: int
        number of frequency samples
    :ivar new_waveform: np.ndarray
        new waveform after spectral filter
    :ivar x_max: float
        chart maximum x value
    :ivar x_min: float
        chart minimum x value
    :ivar y_max: float
        chart maximum y value
    :ivar y_min: float
        chart minimum y value
    """

    def __init__(self):
        super(FreyVoiceEnhancer, self).__init__()

        self.new_waveform = None
        self.flt_spectral = None
        self.fft_n = None
        self.fft_convolve = None
        self.filename = None
        self.flt_band_start = None
        self.flt_band_stop = None
        self.flt_bws = None
        self.frame_size = None
        self.freq_scale = None
        self.output_data = None
        self.spectral_lines = None

        self.y_min = 0.0000001
        self.y_max = 1.0
        self.x_min = 0.0
        self.x_max = 22000.0

    def process(self, filename: str, fft_n: int, flt_bws: list):
        """
        Process an audio file

        Parameters
        ________
        :param fft_convolve:
            whether to convolve the input waveform with the filter output
        :param flt_bws: str:
            tuples of filter cutoff frequencies (lower, upper)
            structure: [(0., 120.), (650., 900.), .., (15000., 16000.)]
        :param filename: str
            path to audio file
        :param fft_n: int
            number of samples to acquire for each fft transform
        """

        self.load(filename)

        if type(fft_n) == list:
            fft_n = fft_n[0]
        self.fft_n = fft_n

        self.frame_size = self.fft_n * self.sampling_period
        self.spectral_lines = int(round((self.fft_n / 2) + 1))
        self.freq_scale = np.linspace(0, self.band_width, self.spectral_lines)

        print(f'Filename:           {self.filename}\n')
        print(f'Total samples:      {self.n_samples}')
        print(f'Sample rate:        {self.sampling_rate}Hz')
        print(f'Sampling period:    {self.sampling_period}s')
        print(f'Block size:         {self.fft_n}')
        print(f'Frame size:         {self.frame_size}s\n')

        self.flt_bws = self.get_nearest_inclusive_frequencies(flt_bws)

        print('\nGenerating frequency-domain filter..')

        self.flt_spectral = np.zeros(shape=(int(self.fft_n / 2),))
        for (lower, upper) in self.flt_bws:
            self.flt_spectral[lower:upper] = 1.

        weights = [.05, .1, .15, .20, .20, .15, .1, .05]
        self.flt_spectral = np.convolve(self.flt_spectral, np.array(weights)[::-1], 'same')
        self.new_waveform = np.ndarray([])

        print('Generating time-domain filter..')

        flt_waveform = []
        for block_index in np.arange(0, self.n_samples-1, self.fft_n):
            if block_index % self.fft_n == 0:
                print(f'Block {block_index}..', end='\r')
            try:
                fft_block = self.process_block(block_index)
                flt_waveform.append(fft_block)
            except ValueError as e:
                print(f'error: {e}')

        # Take rfft of original waveform
        print('Acquiring FFT from input waveform..')

        self.waveform = np.nan_to_num(self.waveform, nan=self.y_min, neginf=self.y_min, posinf=self.y_max)
        self.waveform = minmax_scale(self.waveform, (-1., 1.))
        self.new_waveform = np.nan_to_num(self.new_waveform, nan=self.y_min, neginf=self.y_min, posinf=self.y_max)
        self.new_waveform = minmax_scale(self.new_waveform, (-1., 1.))
        self.waveform = self.waveform[:self.new_waveform.size]

        diff = np.mean(self.new_waveform) - np.mean(self.waveform)
        self.new_waveform = np.subtract(self.new_waveform, diff)

        fft_wav = rfft(self.new_waveform)

        print('Acquiring output waveform from inverse FFT..')
        res = irfft(fft_wav)

        # Extrapolate filter signal to number of samples in rfft
        print('Interpolating time-domain filter..')

        flt = np.interp(np.linspace(0, len(flt_waveform)-1, num=(fft_wav.size-1) * 2),
                        np.arange(len(flt_waveform)), flt_waveform)

        print('Applying time-domain filter..')

        res = minmax_scale(res * flt, (0., 1.))

        self.save(res)

    def get_nearest_inclusive_frequencies(self, cutoff_freqs):
        print('Filter bands selected based on input:')

        fft_freqs = rfftfreq(n=self.fft_n - 1, d=self.sampling_period).astype(dtype=float)

        result = []
        for i in range(0, len(cutoff_freqs)):
            (lower, upper) = cutoff_freqs[i].split('-')

            start = None
            stop = None

            for k, v in np.ndenumerate(fft_freqs):
                if v >= float(lower):
                    start = k[0] - 1
                    break

            for k, v in np.ndenumerate(fft_freqs):
                if v >= float(upper):
                    stop = k[0]
                    break

            # make bands inclusive, i.e. bandwidth encompassed input frequencies
            if start == stop == 1:
                start = 0

            start = max(0, start)
            stop = min(stop, fft_freqs.size-1)

            print(f'- Filter {i+1}: closest inclusive lower cutoff frequency to {lower}Hz: {fft_freqs[start]}Hz')
            print(f'- Filter {i+1}: closest inclusive upper cutoff frequency to {upper}Hz: {fft_freqs[stop]}Hz')

            result.append([start, stop])

        return result

    def process_block(self, block_index: int):
        """
        Process block at block_index

        Parameters
        ________
        :param block_index:
            Index of next block in waveform to process
        :return float
        """
        if block_index + self.fft_n > self.n_samples:
            chunk = self.waveform[block_index:self.n_samples - 1]
        else:
            chunk = self.waveform[block_index:block_index + self.fft_n - 1]

        fft_data = rfft(chunk)  # type: np.ndarray
        fft_conv = self.flt_spectral * fft_data
        self.new_waveform = np.append(self.new_waveform, irfft(fft_conv[1:]))

        # Real part of FFT scaled to y_min-y_max (see __init__)
        fft_lin = np.abs(fft_data)
        fft_lin = minmax_scale(fft_lin, (self.y_min, self.y_max))
        fft_lin = np.nan_to_num(fft_lin, nan=self.y_min, posinf=self.y_max, neginf=self.y_min)

        # Max of the values in the bins that contain the signal most similar to speech
        fft_max = max(map(lambda l: np.max(fft_lin[l[0]:l[1]]), self.flt_bws))

        return fft_max
