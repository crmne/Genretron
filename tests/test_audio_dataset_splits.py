from genretron.audio_dataset import AudioDataset
import numpy

def setup():
    global train, test, valid
    params = {
        'path': '${PYLEARN2_DATA_PATH}/GTZAN',
        'space': 'vector',
        'seconds': 5.0,
        'seed': 1234,
        'balanced_splits': True,
        'preprocessor': 'znormalizer',
        'use_whole_song': True,
        'print_params': False
    }
    params['which_set'] = 'train'
    train = AudioDataset(**params)
    train.process()
    del train.data_x, train.data_y

    params['which_set'] = 'test'
    test = AudioDataset(**params)
    test.process()
    del test.data_x, test.data_y

    params['which_set'] = 'valid'
    valid = AudioDataset(**params)
    valid.process()
    del valid.data_x, valid.data_y

def test_tracks_dont_appear_twice_in_the_same_split():
    for split in [train, test, valid]:
        yield check_tracks_dont_appear_twice_in_the_same_split, split

def check_tracks_dont_appear_twice_in_the_same_split(split):
    assert numpy.all(numpy.unique(split.set_tracks) == numpy.sort(split.set_tracks))

def test_tracks_dont_appear_twice_in_different_splits():
    all_tracks = numpy.sort(numpy.append(train.set_tracks, numpy.append(test.set_tracks, valid.set_tracks)))
    assert numpy.all(numpy.unique(all_tracks) == all_tracks)

def test_indexes_dont_appear_twice_in_the_same_split():
    for split in [train, test, valid]:
        yield check_indexes_dont_appear_twice_in_the_same_split, split

def check_indexes_dont_appear_twice_in_the_same_split(split):
    assert numpy.all(numpy.unique(split.set_indexes) == numpy.sort(split.set_indexes))

def test_indexes_dont_appear_twice_in_different_splits():
    all_indexes = numpy.sort(numpy.append(train.set_indexes, numpy.append(test.set_indexes, valid.set_indexes)))
    assert numpy.all(numpy.unique(all_indexes) == all_indexes)
