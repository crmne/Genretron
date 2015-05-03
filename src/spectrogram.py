import numpy


class Spectrogram(object):
    windows = {
        'square': numpy.ones,
        'hamming': numpy.hamming,
        'hanning': numpy.hanning,
        'bartlett': numpy.bartlett,
        'blackman': numpy.blackman
    }

    _default_window_size = 1024
    _default_window_type = 'square'
    _default_fft_resolution = 1024

    @staticmethod
    def shape(wins, bins):
        return len(wins), bins

    @staticmethod
    def wins(window_size, nframes, step_size):
        window_size = Spectrogram._default_window_size \
            if window_size is None else window_size
        step_size = window_size / 2 if step_size is None else step_size
        return range(window_size, int(nframes), step_size)

    @staticmethod
    def bins(fft_resolution):
        fft_resolution = Spectrogram._default_fft_resolution \
            if fft_resolution is None else fft_resolution
        return (fft_resolution / 2) + 1

    def __init__(self, spectrogram, window_size, step_size, window_type,
                 fft_resolution, wins, bins, nframes):
        self.__dict__.update(locals())
        del self.self

    @classmethod
    def from_waveform(cls, frames, window_size, step_size, window_type,
                      fft_resolution):
        window_size = Spectrogram._default_window_size \
            if window_size is None else window_size
        step_size = window_size / 2 if step_size is None else step_size
        fft_resolution = Spectrogram._default_fft_resolution \
            if fft_resolution is None else fft_resolution
        window_type = Spectrogram._default_window_type \
            if window_type is None else window_type
        window = Spectrogram.windows[window_type](window_size)

        wins = Spectrogram.wins(window_size, len(frames), step_size)
        bins = Spectrogram.bins(fft_resolution)

        spectrogram = numpy.zeros(Spectrogram.shape(wins, bins))

        for i, n in enumerate(wins):
            xseg = frames[n - window_size:n]
            z = numpy.fft.fft(window * xseg, fft_resolution)
            # adding a small quantity fixes log(0) problem
            spectrogram[i, :] = numpy.log(numpy.abs(z[:bins] + 1e-8))

        return cls(spectrogram, window_size, step_size, window_type,
                   fft_resolution, wins, bins, len(frames))

    # def aggregate_features(self, window_size, step_size):
    #     """
    #     Calculate the texture windows as described in Tzanetakis and Cook 2002

    #     Parameters
    #     ----------

    #     window_size: the window size in fft windows
    #     step_size: the stride in fft windows
    #     """
    #     # get the number of frames
    #     n_frames = self.spectrogram.shape[0]
    #     wins = Spectrogram.wins(window_size, n_frames, step_size)
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

    def plot(self, sample_rate=None, title='', with_colorbar=False):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.suptitle('Spectrogram', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.set_title(title)

        if sample_rate is None:
            horizontal_max = len(self.wins)
            ax.set_xlabel('Windows')
            vertical_max = self.bins
            ax.set_ylabel('Bins')
        else:
            horizontal_max = float(self.nframes) / sample_rate
            ax.set_xlabel('Seconds')
            vertical_max = sample_rate / 1 / 2.205
            ax.set_ylabel('Frequency (Hz)')

        cax = ax.imshow(
            self.spectrogram.T,
            interpolation='nearest',
            origin='lower',
            aspect='auto',
            extent=[0, horizontal_max, 0, vertical_max]
        )
        if with_colorbar:
            fig.colorbar(cax)
        plt.show()
