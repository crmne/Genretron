from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import string_utils
from audio_dataset import AudioDataset

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"

# TODO LIST:
# 1. frame as example
# 2. inverted spectrograms
# 3. mfcc
# 4. texture windows


class GTZAN(AudioDataset):
    def __init__(self, which_set,
                 path="${PYLEARN2_DATA_PATH}/GTZAN",
                 feature="spectrogram",
                 preprocess=True,
                 seconds=29.0,
                 window_size=1024,
                 window_type='square',
                 step_size=None,
                 fft_resolution=1024,
                 seed=1234,
                 n_folds=4,
                 run_n=0):
        path = string_utils.preprocess(path)
        super(GTZAN, self).__init__(path)

        valid_features = {
            "spectrogram": self.get_spectrogram_data,
            "inv_spectrogram": self.get_spectrogram_data,
            "signal": self.get_signal_data
        }
        self.params_filter.append('valid_features')

        step_size = step_size if step_size is not None else window_size / 2

        self.__dict__.update(locals())
        del self.self

        print(self)

        if preprocess:
            all_tracks = self.get_indexes('all', n_folds, run_n, seed)
            self.data_x, self.data_y = \
                valid_features[feature](
                    all_tracks, seconds, window_size, step_size,
                    window_type, fft_resolution
                )
            self.scale(self.data_x)
            set_tracks = self.get_indexes(which_set, n_folds, run_n, seed)
            self.data_x, self.data_y = \
                self.filter_indexes(set_tracks, self.data_x, self.data_y)
        else:
            set_tracks = self.get_indexes(which_set, n_folds, run_n, seed)
            self.data_x, self.data_y = \
                valid_features[feature](
                    all_tracks, seconds, window_size, step_size,
                    window_type, fft_resolution
                )


class GTZAN_On_Memory(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, axes=('b', 0, 1, 'c'), **kwargs):
        gtzan = GTZAN(**kwargs)
        view_converter = dense_design_matrix.DefaultViewConverter(
            (gtzan.bins_per_track, gtzan.wins_per_track, 1), axes
        )
        super(GTZAN_On_Memory, self).__init__(
            X=gtzan.data_x,
            y=gtzan.data_y,
            view_converter=view_converter
        )
