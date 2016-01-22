# -*- coding: utf-8 -*-
from librosa.core import stft
from librosa.core import istft
from librosa.display import specshow
import numpy
from scipy.ndimage.interpolation import affine_transform

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
                      fft_resolution=None):
        fft_resolution = Spectrogram.default_fft_resolution if fft_resolution \
            is None else fft_resolution
        step_size = fft_resolution / 2 if step_size is None else step_size

        spectrogram = numpy.log(stft(frames,
                                     hop_length=step_size,
                                     n_fft=fft_resolution) +
                                1e-38)

        assert numpy.isfinite(spectrogram).all

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
                     1e-38,
                     hop_length=self.step_size)

    def scale(self, scale_factors=(1, 1)):
        x_scale_factor, y_scale_factor = scale_factors
        transformation_matrix = numpy.diag(scale_factors)
        data = numpy.real(self.data)
        scaled = affine_transform(data, transformation_matrix)
        return scaled[:scaled.shape[0] / x_scale_factor,
                      :scaled.shape[1] / y_scale_factor]

    def plot(self, sample_rate=22050, title='', with_colorbar=True,
             out=None):
        import matplotlib.pyplot as plt
        plt.title(title)
        specshow(numpy.real(self.data),
                 sr=sample_rate,
                 hop_length=self.step_size,
                 x_axis='time',
                 y_axis='linear',
                 cmap='spectral')
        if with_colorbar:
            plt.colorbar(format='%+2.0f dB',
                         cmap='spectral')
        if out is None:
            plt.show()
        else:
            plt.savefig(out, dpi=300)
        plt.close()
