#!/usr/bin/env python
import numpy
import argparse
import utils
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
from audio_track import AudioTrack
from gtzan import GTZAN


def predict(model_path, track_paths, verbose=False, extended=False):
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
        'window_size', 'step_size', 'window_type', 'fft_resolution'
    ]
    spectrogram_params = \
        {k: dataset_params.get(k, None) for k in valid_spectrogram_params}
    invalid_dataset_params = ['which_set', 'verbose', 'print_params']
    for invalid_key in invalid_dataset_params:
        if dataset_params.get(invalid_key, False):
            del dataset_params[invalid_key]
    dataset = GTZAN(which_set='all',
                    verbose=verbose,
                    print_params=False,
                    **utils.filter_null_args(**dataset_params))

    def space_converter(x):
        return dataset.spaces_converters[dataset.converter]((x, []))[0]

    def view_converter(x):
        vc = dataset.view_converters[dataset.converter]
        if vc is None:
            return x
        else:
            return dataset.view_converters[dataset.converter].\
                design_mat_to_topo_view(x)

    genres = dataset.genres

    for track_path in track_paths:
        if extended:
            track = AudioTrack(track_path)
            y = []
            for offset_seconds in numpy.arange(0, track.seconds_total, seconds):
                if offset_seconds + seconds <= track.seconds_total:
                    track = AudioTrack(track_path, seconds=seconds, offset_seconds=offset_seconds)
                    y.append(track_predict(dataset, track, f, spectrogram_params, view_converter, space_converter))
            y = numpy.array(y).mean(axis=0)
        else:
            track = AudioTrack(track_path, seconds=seconds)
            y = track_predict(dataset, track, f, spectrogram_params, view_converter, space_converter)

        predictions = sorted(
            [(genres[i], v * 100) for i, v in enumerate(y)],
            key=lambda tup: tup[1],
            reverse=True
        )

        print("{}:".format(track.path))
        for prediction in predictions:
            print("{:>10}: {:12.8f}%".format(prediction[0], prediction[1]))


def track_predict(dataset,
                  track,
                  f,
                  spectrogram_params,
                  view_converter,
                  space_converter):

    # calc spectrogram
    spectrogram = track.calc_spectrogram(
        **utils.filter_null_args(**spectrogram_params)).data
    # add the batch size
    spectrogram = spectrogram.reshape(
         (1, spectrogram.shape[0], spectrogram.shape[1]))

    # convert to the format used in training
    x = view_converter(space_converter(spectrogram))

    # get the results
    if dataset.space == "conv2d":
        return f(x)[0]
    elif dataset.space == "vector":
        return f(x).mean(axis=0)
    else:
        raise NotImplementedError


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('tracks', nargs='*')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-e', '--extended', action='store_true')
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    predict(args.model, args.tracks, args.verbose, args.extended)
