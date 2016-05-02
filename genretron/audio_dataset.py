# -*- coding: utf-8 -*-
import os
import numpy
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import string_utils
from theano import config as theanoconfig
from .audio_track import AudioTrack
from .kfold import KFold
from .preprocessors import preprocessor_factory
import utils
import librosa

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class AudioDataset(object):

    """Represent an abstract Audio Dataset"""

    params_filter = [
        'params_filter', 'tracks', 'data_x', 'data_y', 'feature_extractors',
        'spaces_converters', 'index_converters', 'view_converters',
        'view_converter', 'rng'
    ]

    def __init__(self,
                 path,
                 which_set,
                 feature="spectrogram",
                 space="conv2d",
                 axes=('b', 0, 1, 'c'),
                 scale_factors=None,
                 balanced_splits=False,
                 use_whole_song=False,
                 preprocessor=None,
                 seconds=None,
                 step_size=None,
                 fft_resolution=None,
                 seed=None,
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
        tracks, genres = self.tracks_and_genres(path, seconds, use_whole_song)
        assert tracks
        samplerate = tracks[0].samplerate
        seconds = tracks[0].seconds

        # extract parameters from spectrogram
        spec = tracks[0].calc_spectrogram(
            **utils.filter_null_args(step_size=step_size,
                                     fft_resolution=fft_resolution,
                                     scale_factors=scale_factors))
        step_size = spec.step_size
        fft_resolution = spec.fft_resolution
        del spec

        if feature != "signal":
            tracks[0].calc_spectrogram(
                **utils.filter_null_args(
                    step_size=step_size,
                    fft_resolution=fft_resolution,
                    scale_factors=scale_factors
                ))
            bins_per_track, wins_per_track = tracks[0].spectrogram.data.shape

        view_converters = {
            "conv2d": dense_design_matrix.DefaultViewConverter(
                (bins_per_track, wins_per_track, 1), axes),
            "vector": dense_design_matrix.DefaultViewConverter(
                (bins_per_track, 1, 1), axes),
            "signal": None
        }

        view_converter = view_converters[converter]

        preprocessor = None if preprocessor == "None" else preprocessor

        rng = make_np_rng(None, seed, which_method="shuffle")

        self.__dict__.update(locals())
        del self.self

        if print_params:
            print(self)

    def process(self):
        if self.preprocessor is not None:
            # preprocess all the tracks
            all_tracks = self.get_track_ids('all')
            self.data_x, self.data_y = \
                self.spaces_converters[self.converter](
                    self.feature_extractors[
                        self.feature](all_tracks)
                )
            self.preprocess(self.data_x)
            # select only the tracks in the set
            self.set_tracks = self.get_track_ids(self.which_set)
            self.set_indexes = self.index_converters[self.converter](self.set_tracks)
            self.data_x, self.data_y = \
                self.filter_indexes(self.set_indexes, self.data_x, self.data_y)
        else:
            self.set_tracks = self.get_track_ids(self.which_set)
            self.data_x, self.data_y = \
                self.spaces_converters[self.converter](
                    self.feature_extractors[
                        self.feature](self.set_tracks)
                )

    def __str__(self):
        from pprint import pformat
        return pformat(utils.filter_keys_from_dict(self.params_filter,
                                                   self.__dict__))

    def __repr__(self):
        return "".join([self.__module__, ".", self.__class__.__name__, "(**",
                        self.__str__(), ")"])

    def get_signal_data(self, indexes):
        data_x = numpy.zeros(
            (len(indexes), self.seconds * self.samplerate),
            dtype=numpy.dtype(theanoconfig.floatX).type)
        data_y = numpy.zeros(
            (len(indexes), len(self.genres)),
            dtype=numpy.int8)
        for data_i, index in enumerate(indexes):
            track = self.tracks[index]
            data_x[data_i] = track.signal
            data_y[data_i][self.genres.index(track.genre)] = 1
            track.rm_signal()
        return data_x, data_y

    def get_spectrogram_data(self, indexes):
        data_x = numpy.zeros(
            (len(indexes), self.bins_per_track, self.wins_per_track),
            dtype=numpy.dtype(theanoconfig.floatX).type)
        data_y = numpy.zeros(
            (len(indexes), len(self.genres)),
            dtype=numpy.int8)

        for data_i, index in enumerate(indexes):
            track = self.tracks[index]
            if self.verbose:
                print("calculating spectrogram of " + track.path)
            data_x[data_i], _ = librosa.magphase(track.calc_spectrogram(
                **utils.filter_null_args(
                    step_size=self.step_size,
                    fft_resolution=self.fft_resolution,
                    scale_factors=self.scale_factors
                )
            ).data)
            data_y[data_i][self.genres.index(track.genre)] = 1
            track.rm_spectrogram()
            track.rm_signal()
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

    def get_inv_spectrogram_data(self, *args):
        if self.verbose:
            print("transposing spectrograms...")
        return AudioDataset.invert_wins_bins(self.get_spectrogram_data(*args))

    def filter_indexes(self, indexes, data_x, data_y):
        if self.verbose:
            print("filtering indexes...")
        return numpy.take(data_x, indexes, axis=0), \
            numpy.take(data_y, indexes, axis=0)

    def preprocess(self, data_x):
        if self.verbose:
            print("preprocessing with {0}...".format(self.preprocessor))
        preprocessor = preprocessor_factory(self.preprocessor)
        preprocessor.fit_transform(data_x)

    def tracks_and_genres(self, audio_folder, seconds, use_whole_song):
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
                        track = AudioTrack(filename,
                                           genre=genre,
                                           seconds=seconds)
                        tracks.append(track)
                        genres.add(genre)
                        self.ntracksegments = 1
                        if use_whole_song:
                            self.ntracksegments = int(
                                track.seconds_total / seconds)
                            for i in range(1, self.ntracksegments):
                                tracks.append(
                                    AudioTrack(filename,
                                               genre=genre,
                                               seconds=seconds,
                                               offset_seconds=seconds * i))
        return tracks, sorted(genres)

    def get_all_file_ids(self):
        return numpy.arange(len(self.tracks) / self.ntracksegments)

    def get_all_track_ids(self):
        return numpy.arange(len(self.tracks))

    def get_unbalanced_file_ids(self):
        file_ids = self.get_all_file_ids()
        self.rng.shuffle(file_ids)
        kf = KFold(file_ids, n_folds=self.n_folds)
        return kf.runs[self.run_n][self.which_set]

    def get_file_n_by_genre(self):
        genre_ids = {}
        for file_n in self.get_all_file_ids():
            track_n = file_n * self.ntracksegments
            genre = self.tracks[track_n].genre
            if genre in genre_ids:
                genre_ids[genre].append(file_n)
            else:
                genre_ids[genre] = [file_n]
        return genre_ids

    def get_balanced_file_ids(self):
        genre_ids = self.get_file_n_by_genre()
        file_ids = []
        for genre in self.genres:
            idxs = numpy.asarray(genre_ids[genre])
            self.rng.shuffle(idxs)
            kf = KFold(idxs, n_folds=self.n_folds)
            file_ids.extend(kf.runs[self.run_n][self.which_set])
        self.rng.shuffle(file_ids)
        return file_ids

    def file_ids_to_track_ids(self, file_ids):
        expanded = [numpy.arange(
            i * self.ntracksegments,
            (i * self.ntracksegments) + self.ntracksegments)
            for i in file_ids]
        return numpy.array(expanded).flatten()

    def get_track_ids(self, which_set):
        """
        Returns the indexes of the tracks according to the split they are in,
        making sure that track segments appear in the same split and when
        balanced_splits is True, that the number of tracks for each genre is
        the same across splits.
        """
        if which_set == 'all':
            file_ids = self.get_all_file_ids()
        else:
            if self.balanced_splits:
                file_ids = self.get_balanced_file_ids()
            else:
                file_ids = self.get_unbalanced_file_ids()

        if self.use_whole_song:
            track_ids = self.file_ids_to_track_ids(file_ids)
        else:
            track_ids = file_ids

        return track_ids

    def track_ids_to_frame_ids(self, track_ids):
        return numpy.array([numpy.arange(x * self.wins_per_track, (
            x * self.wins_per_track) + self.wins_per_track) for x in track_ids
        ]).flatten()
