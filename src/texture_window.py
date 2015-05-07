import numpy
from two_dimensional_feature import TwoDimensionalFeature


class TextureWindow(TwoDimensionalFeature):
    default_window_size = 40
    default_window_type = 'square'

    def __init__(self,
                 matrix,
                 window_size,
                 step_size,
                 window_type,
                 wins,
                 bins):
        super(TextureWindow, self).__init__(
            matrix,
            window_size,
            step_size,
            window_type,
            wins,
            bins
        )

    @classmethod
    def from_matrix(cls, matrix,
                    window_size=default_window_size,
                    step_size=None,
                    window_type=default_window_type,):
        step_size = window_size / 2 if step_size is None else step_size

        wins = cls.wins(matrix.shape[0], window_size, step_size)
        bins = matrix.shape[1]

        window = cls.window_types[window_type]((window_size, bins))

        mean = numpy.zeros(cls.shape(wins, bins))
        variance = numpy.zeros(cls.shape(wins, bins))

        for i, n in enumerate(wins):
            xseg = matrix[n - window_size:n]
            mean[i, :] = (window * xseg).mean(axis=0)
            variance[i, :] = (window * xseg).var(axis=0)

        return cls(mean, window_size, step_size, window_type, wins, bins)
