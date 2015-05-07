import numpy
from two_dimensional_feature import TwoDimensionalFeature


class Spectrogram(TwoDimensionalFeature):
    default_fft_resolution = 1024

    @staticmethod
    def bins(fft_resolution=None):
        fft_resolution = fft_resolution or Spectrogram.default_fft_resolution
        return (fft_resolution / 2) + 1

    def __init__(self, spectrogram, window_size, step_size, window_type,
                 fft_resolution, wins, bins, nframes):
        super(Spectrogram, self).__init__(
            spectrogram,
            window_size,
            step_size,
            window_type,
            wins,
            bins
        )
        self.fft_resolution = fft_resolution
        self.nframes = nframes

    @classmethod
    def from_waveform(cls, frames,
                      window_size=TwoDimensionalFeature.default_window_size,
                      step_size=None,
                      window_type=TwoDimensionalFeature.default_window_type,
                      fft_resolution=default_fft_resolution):
        step_size = window_size / 2 if step_size is None else step_size
        window = cls.window_types[window_type](window_size)

        wins = cls.wins(len(frames), window_size, step_size)
        bins = cls.bins(fft_resolution)

        spectrogram = numpy.zeros(cls.shape(wins, bins))

        for i, n in enumerate(wins):
            xseg = frames[n - window_size:n]
            z = numpy.fft.fft(window * xseg, fft_resolution)
            # adding a small quantity fixes log(0) problem
            spectrogram[i, :] = numpy.log(numpy.abs(z[:bins] + 1e-8))

        return cls(spectrogram, window_size, step_size, window_type,
                   fft_resolution, wins, bins, len(frames))
