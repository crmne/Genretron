import stft
import numpy
import math

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class Spectrogram():
    window_types = {
        'square': numpy.ones,
        'hamming': numpy.hamming,
        'hanning': numpy.hanning,
        'bartlett': numpy.bartlett,
        'blackman': numpy.blackman
    }

    default_window_type = 'square'
    default_fft_resolution = 1024

    @staticmethod
    def shape(wins, bins):
        return wins, bins

    @classmethod
    def wins(cls, nframes, fft_resolution=None, step_size=None):
        fft_resolution = cls.default_fft_resolution \
            if fft_resolution is None else fft_resolution
        step_size = fft_resolution / 2 if step_size is None else step_size
        return int(math.ceil(nframes / fft_resolution) * fft_resolution - nframes)

    @staticmethod
    def bins(fft_resolution=None):
        fft_resolution = fft_resolution or Spectrogram.default_fft_resolution
        return (fft_resolution / 2) + 1

    @classmethod
    def from_waveform(cls, frames,
                      step_size=None,
                      window_type=default_window_type,
                      fft_resolution=default_fft_resolution):
        step_size = fft_resolution / 2 if step_size is None else step_size
        window = cls.window_types[window_type]

        spectrogram = stft.spectrogram(frames,
                                       framelength=fft_resolution,
                                       hopsize=step_size,
                                       window=window)

        bins = spectrogram.shape[0]
        wins = spectrogram.shape[1]

        return cls(spectrogram, step_size, window_type,
                   fft_resolution, wins, bins, len(frames))

    def __init__(self, data, step_size, window_type,
                 fft_resolution, wins, bins, nframes):
        self.__dict__.update(locals())
        del self.self

    def to_signal(self):
        return stft.ispectrogram(self.data)

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
            extent=[0, horizontal_max, 0, vertical_max]
        )
        if with_colorbar:
            fig.colorbar(cax)
        plt.show()
