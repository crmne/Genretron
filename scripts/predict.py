#!/usr/bin/env python
import argparse
import collections
import numpy

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
from audio_track import AudioTrack
from gtzan import GTZAN


def predict(model_path, track_paths, verbose=False):
    """
    Predict from a pkl file.

    Parameters
    ----------
    model_path : str
        The file name of the model file.
    track_paths : str
        The file name of the file to test/predict.
    """

    if verbose:
        print("loading model...")

    try:
        model = serial.load(model_path)
    except Exception as e:
        print("error loading {}:".format(model_path))
        print(e)
        return False

    if verbose:
        print("setting up symbolic expressions...")

    X = model.get_input_space().make_theano_batch()
    y = model.fprop(X)

    f = function([X], y, allow_input_downcast=True)

    if verbose:
        print("loading data and predicting...")

    # load dataset parameters
    dataset_params = yaml_parse.load(
        model.dataset_yaml_src, instantiate=False).keywords
    seconds = dataset_params.get('seconds', None)
    valid_spectrogram_params = [
            'window_size', 'step_size', 'window_type', 'fft_resolution']
    spectrogram_params = \
        {k: dataset_params.get(k, None) for k in valid_spectrogram_params}
    dataset = GTZAN(which_set='all',
                    seconds=seconds,
                    print_params=verbose,
                    verbose=verbose,
                    **spectrogram_params)

    def space_converter(x):
        return dataset.spaces_converters[dataset.converter]((x, []))[0]

    def view_converter(x):
        return dataset.view_converters[dataset.converter].\
            design_mat_to_topo_view(x)

    genres = dataset.genres

    for track_path in track_paths:
        # load track
        track = AudioTrack(track_path, seconds=seconds)
        # calc spectrogram
        spectrogram = track.calc_spectrogram(**spectrogram_params).spectrogram
        # add the batch size
        spectrogram = spectrogram.reshape(
             (1, spectrogram.shape[0], spectrogram.shape[1]))

        # convert to the format used in training
        x = view_converter(space_converter(spectrogram))

        y = {genres[index]: "{}%".format(round(value * 100, 4)) for index, value in enumerate(f(x)[0])}

        # print(collections.Counter(y).most_common())
        from pprint import pformat
        print("{}:\n{}".format(track.path, pformat(y)))


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('tracks', nargs='*')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    predict(args.model, args.tracks, args.verbose)
