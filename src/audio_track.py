import os
import numpy
import theano.config
from scikits.audiolab import Sndfile
from scikits.audiolab import play
from scikits.audiolab import available_file_formats
from spectrogram import Spectrogram


class AudioTrack(object):
    extensions = available_file_formats()

    def __init__(self, path, genre=None, seconds=None):
        super(AudioTrack, self).__init__()

        filename = os.path.basename(path)
        _seconds = seconds

        self.__dict__.update(locals())
        del self.self
        del seconds

    @property
    def samplerate(self):
        return Sndfile(self.path, mode='r').samplerate

    @property
    def channels(self):
        return Sndfile(self.path, mode='r').channels

    @property
    def nframes_total(self):
        return Sndfile(self.path, mode='r').nframes

    @property
    def nframes(self):
        return self.seconds * self.samplerate

    @property
    def format(self):
        return Sndfile(self.path, mode='r').format

    @property
    def encoding(self):
        return Sndfile(self.path, mode='r').encoding

    @property
    def seconds_total(self):
        return float(self.nframes_total) / float(self.samplerate)

    @property
    def seconds(self):
        if self._seconds is None:
            self._seconds = self.seconds_total
        return self._seconds

    @property
    def signal(self):
        if not hasattr(self, '_signal'):
            self._signal = Sndfile(self.path, mode='r').read_frames(
                self.nframes,
                dtype=theano.config.floatX
            )
        return self._signal

    @property
    def spectrogram(self):
        return self.calc_spectrogram()

    def calc_spectrogram(self,
                         window_size=None,
                         step_size=None,
                         window_type=None,
                         fft_resolution=None):
        if not hasattr(self, '_spectrogram'):
            self._spectrogram = Spectrogram.from_waveform(
                self.signal,
                window_size,
                step_size,
                window_type,
                fft_resolution
            )
        return self._spectrogram

    def play(self):
        play(self.signal, fs=self.samplerate)

    def plot_signal(self, title=None):
        import matplotlib.pyplot as plt
        if title is None:
            title = self.filename
        plt.title(title)
        plt.plot(
            numpy.linspace(0, self.seconds, num=self.nframes),
            self.signal
        )
        plt.xlim(0, self.seconds)
        plt.show()

    def plot_spectrogram(self, title=None):
        if title is None:
            title = self.filename
        self.spectrogram.plot(title=title)
