# -*- coding: utf-8 -*-
from librosa.core import stft
from librosa.core import istft
import numpy

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class Spectrogram():
    default_fft_resolution = 1024

    @classmethod
    def from_waveform(cls, frames,
                      step_size=None,
                      fft_resolution=default_fft_resolution):
        step_size = fft_resolution / 2 if step_size is None else step_size

        spectrogram = numpy.log(stft(frames,
                                     hop_length=step_size,
                                     n_fft=fft_resolution) +
                                numpy.finfo('float32').min)

        bins = spectrogram.shape[0]
        wins = spectrogram.shape[1]

        return cls(spectrogram, step_size,
                   fft_resolution, wins, bins, len(frames))

    def __init__(self, data, step_size,
                 fft_resolution, wins, bins, nframes):
        self.__dict__.update(locals())
        del self.self

    def to_signal(self):
        return istft(numpy.exp(self.data) -
                     numpy.finfo('float32').min,
                     hop_length=self.step_size)

    def plot(self, sample_rate=None, title='', with_colorbar=False):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.suptitle(self.__class__.__name__, fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.set_title(title)

        if sample_rate is None:
            horizontal_max = self.wins
            ax.set_xlabel('Windows')
            vertical_max = self.bins
            ax.set_ylabel('Bins')
        else:
            horizontal_max = float(self.nframes) / sample_rate
            ax.set_xlabel('Seconds')
            vertical_max = sample_rate / 1 / 2.205
            ax.set_ylabel('Frequency (Hz)')

        cax = ax.imshow(
            numpy.real(self.data),
            interpolation='nearest',
            origin='lower',
            aspect='auto',
            extent=[0, horizontal_max, 0, vertical_max],
            cmap='spectral'
        )
        if with_colorbar:
            fig.colorbar(cax)
        plt.show()
