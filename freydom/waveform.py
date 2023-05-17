import audiofile

from os.path import realpath, basename, splitext
from numpy import ndarray


class WaveForm(object):
    """
    WaveformInterface

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

    def __init__(self):
        """
        Constructor
        """

        super(WaveForm, self).__init__()

        self.band_width = None
        self.bit_depth = None
        self.duration = None
        self.filename = None
        self.n_samples = None
        self.sampling_period = None
        self.sampling_rate = None
        self.waveform = None

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

    def save(self, waveform: ndarray):
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


