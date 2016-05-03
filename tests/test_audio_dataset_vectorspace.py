from genretron.audio_dataset import AudioDataset
import numpy

def setup():
    global dataset, data_x, data_y
    params = {
        'path': '${PYLEARN2_DATA_PATH}/GTZAN',
        'space': 'vector',
        'seconds': 5.0,
        'seed': 1234,
        'balanced_splits': True,
        'use_whole_song': True,
        'print_params': False,
        'which_set': 'valid'
    }
    dataset = AudioDataset(**params)
    dataset.set_tracks = dataset.get_track_ids(dataset.which_set)
    data_x, data_y = dataset.feature_extractors[dataset.feature](dataset.set_tracks)

def test_x_to_vectorspace_should_return_data_in_right_order():
    num_examples, column_size, row_size = data_x.shape
    first_column = []
    for i in range(column_size):
        first_column.append(data_x[0][i][0])
    first_column = numpy.array(first_column)
    reshaped = AudioDataset.x_to_vectorspace(data_x)
    assert numpy.all(reshaped[0] == first_column)

def test_y_to_vectorspace_should_return_data_in_right_order():
    pass
