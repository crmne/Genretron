from pylearn2.datasets import dense_design_matrix
from audio_dataset import AudioDataset

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class GTZAN(AudioDataset):
    def __init__(self, path="${PYLEARN2_DATA_PATH}/GTZAN",
                 seconds=29.0, **kwargs):
        super(GTZAN, self).__init__(path, seconds=seconds, **kwargs)


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
