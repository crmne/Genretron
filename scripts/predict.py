#!/usr/bin/env python
import argparse
import collections
import numpy

from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from audio_dataset import AudioDataset
from audio_track import AudioTrack


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('track')
    return parser


def predict(model_path, track_path):
    """
    Predict from a pkl file.

    Parameters
    ----------
    model_path : str
        The file name of the model file.
    track_path : str
        The file name of the file to test/predict.
    """

    print("loading model...")

    try:
        model = serial.load(model_path)
    except Exception as e:
        print("error loading {}:".format(model_path))
        print(e)
        return False

    print("setting up symbolic expressions...")

    X = model.get_input_space().make_theano_batch()
    y = model.fprop(X)

    y = T.argmax(y, axis=1)

    f = function([X], y, allow_input_downcast=True)

    print("loading data and predicting...")

    # x is a numpy array
    spectrogram = AudioTrack(track_path).spectrogram.spectrogram
    x = AudioDataset.x_to_vectorspace(numpy.array([spectrogram]))

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    y = [genres[i] for i in f(x).tolist()]

    print(collections.Counter(y).most_common())

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    predict(args.model, args.track)
