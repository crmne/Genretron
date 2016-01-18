# -*- coding: utf-8 -*-
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

    def __init__(self, **kwargs):
        gtzan = GTZAN(**kwargs)
        gtzan.process()
        super(GTZAN_On_Memory, self).__init__(
            X=gtzan.data_x,
            y=gtzan.data_y,
            view_converter=gtzan.view_converter
        )
