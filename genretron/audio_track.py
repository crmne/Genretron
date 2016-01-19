import os
import numpy
from theano import config as theanoconfig
from scikits.audiolab import Sndfile
from scikits.audiolab import play
from scikits.audiolab import available_file_formats
from spectrogram import Spectrogram

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class AudioTrack(object):
    extensions = available_file_formats()

    def __init__(self, path, genre=None, seconds=None, offset_seconds=None):
        super(AudioTrack, self).__init__()

        filename = os.path.basename(path)
        _seconds = seconds

        self.__dict__.update(locals())
        del self.self
        del seconds

        if self.offset_seconds is not None:
            assert (self.offset_seconds + self.seconds) <= self.seconds_total

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return "".join(
            [self.__module__,
             ".",
             self.__class__.__name__,
             "(**",
             self.__str__(),
             ")"]
        )

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
    def nframes_extended(self):
        return (self.seconds + self.offset_seconds) * self.samplerate

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
            if self.offset_seconds is None:
                self._signal = Sndfile(self.path, mode='r').read_frames(
                    self.nframes,
                    dtype=numpy.dtype(theanoconfig.floatX).type
                )
            else:
                self._signal = Sndfile(self.path, mode='r').read_frames(
                    self.nframes_extended,
                    dtype=numpy.dtype(theanoconfig.floatX).type
                )
                self._signal = \
                    self._signal[self.nframes_extended - self.nframes:]
            self.normalize()
        return self._signal

    @signal.setter
    def signal(self, value):
        self._signal = value

    def rm_signal(self):
        if hasattr(self, '_signal'):
            del self._signal

    def normalize(self):
        self.signal /= numpy.max(numpy.abs(self.signal), axis=0)

    @property
    def spectrogram(self):
        if not hasattr(self, '_spectrogram'):
            return self.calc_spectrogram()
        else:
            return self._spectrogram

    def calc_spectrogram(self, **kwargs):
        self._spectrogram = Spectrogram.from_waveform(
            self.signal,
            **kwargs
        )
        return self._spectrogram

    def rm_spectrogram(self):
        if hasattr(self, '_spectrogram'):
            del self._spectrogram

    def play(self):
        play(self.signal, fs=self.samplerate)

    def plot_signal(self, title=None, out=None):
        import matplotlib.pyplot as plt
        if title is None:
            title = self.filename

        fig = plt.figure()
        fig.suptitle('Signal', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Amplitude')
        ax.plot(
            numpy.linspace(0, self.seconds, num=self.nframes),
            self.signal
        )
        ax.set_xlim(0, self.seconds)
        if out is None:
            plt.show()
        else:
            plt.savefig(out)

    def plot_spectrogram(self, title=None, out=None):
        if title is None:
            title = self.filename
        self.spectrogram.plot(
            sample_rate=self.samplerate,
            title=title,
            with_colorbar=True,
            out=out
        )
