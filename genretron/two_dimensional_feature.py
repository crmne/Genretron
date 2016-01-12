import numpy

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class TwoDimensionalFeature(object):
    window_types = {
        'square': numpy.ones,
        'hamming': numpy.hamming,
        'hanning': numpy.hanning,
        'bartlett': numpy.bartlett,
        'blackman': numpy.blackman
    }

    default_window_size = 1024
    default_window_type = 'square'

    def __init__(self,
                 data,
                 window_size,
                 step_size,
                 window_type,
                 wins,
                 bins):
        nframes = len(data)
        self.__dict__.update(locals())
        del self.self

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
