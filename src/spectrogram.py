import numpy


class Spectrogram(object):
    windows = {
        'square': numpy.ones,
        'hamming': numpy.hamming,
        'hanning': numpy.hanning,
        'bartlett': numpy.bartlett,
        'blackman': numpy.blackman
    }

    @staticmethod
    def shape(wins, bins):
        return len(wins), bins

    @staticmethod
    def wins(win_size, nframes, step_size):
        return range(win_size, int(nframes), step_size)

    @staticmethod
    def bins(fft_resolution):
        return fft_resolution / 2

    def __init__(self, spectrogram):
        self.spectrogram = spectrogram

    @classmethod
    def from_waveform(cls, frames, win_size, step_size, window_type,
                      fft_resolution):
        window = Spectrogram.windows[window_type](win_size)

        wins = Spectrogram.wins(win_size, len(frames), step_size)
        bins = Spectrogram.bins(fft_resolution)

        spectrogram = numpy.zeros(Spectrogram.shape(wins, bins))

        for i, n in enumerate(wins):
            xseg = frames[n - win_size:n]
            z = numpy.fft.fft(window * xseg, fft_resolution)
            # adding a small quantity fixes log(0) problem
            spectrogram[i, :] = numpy.log(numpy.abs(z[:bins] + 1e-8))

        return cls(spectrogram)

    # def aggregate_features(self, win_size, step_size):
    #     """
    #     Calculate the texture windows as described in Tzanetakis and Cook 2002

    #     Parameters
    #     ----------

    #     win_size: the window size in fft windows
    #     step_size: the stride in fft windows
    #     """
    #     # get the number of frames
    #     n_frames = self.spectrogram.shape[0]
    #     wins = Spectrogram.wins(win_size, n_frames, step_size)
    #     bins = self.spectrogram.shape[1]

    #     self.aggregated = numpy.zeros(Spectrogram.shape(wins, bins))

    # def plot_aggregated_features(self):
    #     if not self.aggregated:
    #         return
    #     import matplotlib.pyplot as plt
    #     plt.imshow(
    #         self.aggregated.T,
    #         interpolation='nearest',
    #         origin='lower',
    #         aspect='auto'
    #     )
    #     plt.show()

    # TODO: use time scale
    def plot(self, title="Spectrogram"):
        import matplotlib.pyplot as plt
        plt.title(title)
        plt.imshow(
            self.spectrogram.T,
            interpolation='nearest',
            origin='lower',
            aspect='auto'
        )
        plt.show()
