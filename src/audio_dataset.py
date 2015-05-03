import os
import numpy
from sklearn.preprocessing import StandardScaler
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import string_utils
from theano import config as theanoconfig
from spectrogram import Spectrogram
from kfold import KFold
import utils
from audio_track import AudioTrack

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class AudioDataset(object):
    """Represent an abstract Audio Dataset"""

    params_filter = [
        'params_filter',
        'tracks',
        'data_x',
        'data_y',
        'feature_extractors',
        'spaces_converters',
        'index_converters',
        'view_converters',
        'view_converter',
    ]

    def __init__(self, path, which_set,
                 feature="spectrogram",
                 space="conv2d",
                 axes=('b', 0, 1, 'c'),
                 preprocess=True,
                 seconds=None,
                 window_size=None,
                 window_type=None,
                 step_size=None,
                 fft_resolution=None,
                 seed=1234,
                 n_folds=4,
                 run_n=0,
                 verbose=False,
                 print_params=True):
        super(AudioDataset, self).__init__()

        converter = space

        # signal is 1D
        if feature == "signal":
            space = "vector"
            converter = "signal"

        # inverting a spectrogram if the space is a vector doesn't make sense
        if space == "vector" and feature == "inv_spectrogram":
            feature == "spectrogram"

        feature_extractors = {
            "spectrogram": self.get_spectrogram_data,
            "inv_spectrogram": self.get_inv_spectrogram_data,
            "signal": self.get_signal_data
        }

        spaces_converters = {
            "conv2d": AudioDataset.twod_to_conv2dspaces,
            "vector": AudioDataset.twod_to_vectorspaces,
            "signal": lambda x: x
        }

        index_converters = {
            "conv2d": lambda x: x,
            "vector": self.track_ids_to_frame_ids,
            "signal": lambda x: x
        }

        path = string_utils.preprocess(path)

        # init dynamic params
        tracks, genres = self.tracks_and_genres(path, seconds)
        samplerate = tracks[0].samplerate
        seconds = tracks[0].seconds

        wins_per_track = len(
            Spectrogram.wins(
                window_size,
                seconds * samplerate,
                step_size
            )
        )

        bins_per_track = Spectrogram.bins(fft_resolution)

        view_converters = {
            "conv2d": dense_design_matrix.DefaultViewConverter(
                (bins_per_track, wins_per_track, 1), axes
            ),
            "vector": None,
            "signal": None
        }

        view_converter = view_converters[converter]

        self.__dict__.update(locals())
        del self.self

        if print_params:
            print(self)

    def process(self):
        if self.preprocess:
            all_tracks = self.get_track_ids(
                'all', self.n_folds, self.run_n, self.seed)
            self.data_x, self.data_y = \
                self.spaces_converters[self.converter](
                    self.feature_extractors[self.feature](
                        all_tracks, self.seconds, self.window_size,
                        self.step_size, self.window_type, self.fft_resolution
                    )
                )
            self.scale(self.data_x)
            set_tracks = self.get_track_ids(
                self.which_set, self.n_folds, self.run_n, self.seed)
            set_indexes = self.index_converters[self.converter](set_tracks)
            self.data_x, self.data_y = \
                self.filter_indexes(set_indexes, self.data_x, self.data_y)
        else:
            set_tracks = self.get_track_ids(
                self.which_set, self.n_folds, self.run_n, self.seed)
            self.data_x, self.data_y = \
                self.spaces_converters[self.converter](
                    self.feature_extractors[self.feature](
                        set_tracks, self.seconds, self.window_size,
                        self.step_size, self.window_type, self.fft_resolution
                    )
                )

    def __repr__(self):
        from pprint import pformat
        return "{}(\n{}\n)".format(
            self.__class__.__name__,
            pformat(
                utils.filter_keys_from_dict(
                    self.params_filter,
                    self.__dict__
                    )
                )
            )

    def get_signal_data(self, indexes, seconds, **kwargs):
        data_x = numpy.zeros(
            (len(indexes), seconds * self.samplerate),
            dtype=numpy.dtype(theanoconfig.floatX).type)
        data_y = numpy.zeros(
            (len(indexes), len(self.genres)),
            dtype=numpy.int8)
        for data_i, index in enumerate(indexes):
            track = self.tracks[index]
            data_x[data_i] = track.signal
            data_y[data_i][self.genres.index(track.genre)] = 1
        return data_x, data_y

    def get_spectrogram_data(self, indexes, seconds, window_size,
                             step_size, window_type,
                             fft_resolution):
        data_x = numpy.zeros(
            (len(indexes), self.wins_per_track, self.bins_per_track),
            dtype=numpy.dtype(theanoconfig.floatX).type)
        data_y = numpy.zeros(
            (len(indexes), len(self.genres)),
            dtype=numpy.int8)
        for data_i, index in enumerate(indexes):
            track = self.tracks[index]
            if self.verbose:
                print("calculating spectrogram of " + track.path)
            data_x[data_i] = track.calc_spectrogram(
                window_size, step_size, window_type,
                fft_resolution).spectrogram
            data_y[data_i][self.genres.index(track.genre)] = 1
            del track._signal
        return data_x, data_y

    @staticmethod
    def invert_wins_bins(data):
        return data[0].transpose((0, 2, 1)), data[1]

    @staticmethod
    def x_to_conv2dspace(x):
        return numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

    @staticmethod
    def twod_to_conv2dspaces(data):
        data_x = AudioDataset.x_to_conv2dspace(data[0])
        data_y = data[1]
        return data_x, data_y

    @staticmethod
    def x_to_vectorspace(x):
        return numpy.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))

    @staticmethod
    def y_to_vectorspace(x, y):
        return numpy.repeat(y, x.shape[1], axis=0)

    @staticmethod
    def twod_to_vectorspaces(data):
        data_x = AudioDataset.x_to_vectorspace(data[0])
        data_y = AudioDataset.y_to_vectorspace(data[0], data[1])
        return data_x, data_y

    def get_inv_spectrogram_data(self, **kwargs):
        if self.verbose:
            print("transposing spectrograms...")
        return AudioDataset.invert_wins_bins(
                self.get_spectrogram_data(**kwargs)
            )

    def filter_indexes(self, indexes, data_x, data_y):
        if self.verbose:
            print("filtering indexes...")
        return numpy.take(data_x, indexes, axis=0), \
            numpy.take(data_y, indexes, axis=0)

    def scale(self, data_x):
        if self.verbose:
            print("preprocessing...")
        preprocessor = StandardScaler(copy=False)
        preprocessor.fit_transform(data_x)

    def tracks_and_genres(self, audio_folder, seconds):
        """
        Returns a list of tracks and a list of genres.
        Assumes that audio_folder contains genre-named folders, which
        contain audio files in that genre.
        """
        tracks = []
        genres = set()
        audio_folder = os.path.expanduser(audio_folder)
        for root, dirnames, filenames in os.walk(audio_folder):
            for f in filenames:
                for x in AudioTrack.extensions:
                    if f.endswith(x):
                        filename = os.path.join(root, f)
                        genre = os.path.basename(root)
                        tracks.append(
                            AudioTrack(filename, genre=genre, seconds=seconds)
                        )
                        genres.add(genre)
        return tracks, sorted(genres)

    def get_track_ids(self, set, nfolds, run_n, seed):
        """
        Returns the indexes of the tracks if the space is Conv2D,
        the indexes of the frames if the space is Vector
        """
        track_ids = numpy.arange(len(self.tracks))
        if set != 'all':
            rng = make_np_rng(None, seed, which_method="shuffle")
            rng.shuffle(track_ids)
            kf = KFold(track_ids, n_folds=nfolds)
            track_ids = kf.runs[run_n][set]
        return track_ids

    def track_ids_to_frame_ids(self, track_ids):
        return numpy.array(
                    [numpy.arange(
                        x * self.wins_per_track,
                        (x * self.wins_per_track) + self.wins_per_track
                    ) for x in track_ids]).flatten()
