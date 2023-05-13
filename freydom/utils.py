
import audiofile
import numpy as np
from os.path import basename, splitext


class AudioFile(object):

    def __init__(self, filename):
        super(AudioFile, self).__init__()

        self.filename = filename

        self.bit_depth = None
        self.channels = None
        self.duration = None
        self.sampling_rate = None
        self.signal = None

        self._load()

    def _load(self):
        self.signal, self.sampling_rate = audiofile.read(self.filename, always_2d=True)
        self.channels = self.signal.shape[0]
        self.bit_depth = audiofile.bit_depth(self.filename)
        self.duration = audiofile.duration(self.filename)

    def update(self, new_signal: np.ndarray):
        if new_signal.shape is not self.signal.shape:
            raise RuntimeError(
                f'New signal shape ({new_signal.shape}) does not match old signal shape ({str(self.signal.shape)})')

        self.signal = new_signal

    def save(self):
        filename = f'{basename(self.filename)}-processed.{splitext(self.filename)[1]}'
        audiofile.write(filename, self.signal, self.sampling_rate, self.bit_depth, True)
