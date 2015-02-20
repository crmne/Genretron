import os
import numpy
from collections import OrderedDict
from scikits.audiolab import Sndfile
from scikits.audiolab import available_file_formats
from sklearn.preprocessing import StandardScaler
from pylearn2.utils.rng import make_np_rng
from spectrogram import Spectrogram
from kfold import KFold
import utils

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class AudioDataset(object):
    """Represent an abstract Audio Dataset"""

    params_filter = [
        'params_filter',
        'files_and_genres',
        'data_x',
        'data_y'
    ]

    def __init__(self, path):
        super(AudioDataset, self).__init__()

        # init dynamic params
        files_and_genres = self.list_audio_files_and_genres(path)
        number_of_tracks = len(files_and_genres)
        genres = numpy.unique(files_and_genres.values()).tolist()
        first_song = next(iter(files_and_genres))
        sample_rate = AudioDataset.read_sample_rate(first_song)
        channels = AudioDataset.read_channels(first_song)
        del first_song

        self.__dict__.update(locals())
        del self.self

    @staticmethod
    def read_sample_rate(filename):
        sndfile = Sndfile(filename, mode='r')
        return sndfile.samplerate

    @staticmethod
    def read_channels(filename):
        sndfile = Sndfile(filename, mode='r')
        return sndfile.channels

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
            (len(indexes), seconds * self.sample_rate),
            dtype=numpy.float32)
        data_y = numpy.zeros(
            (len(indexes), len(self.genres)),
            dtype=numpy.int8)
        for data_i, index in enumerate(indexes):
            filename = self.files_and_genres[index][0]
            genre = self.files_and_genres[index][1]
            data_x[data_i] = self.read_raw_audio(filename)
            data_y[data_i][self.genres.index(genre)] = 1
        return data_x, data_y

    def get_spectrogram_data(self, indexes, seconds, window_size=1024,
                             step_size=512, window_type='square',
                             fft_resolution=1024):
        self.wins_per_track = len(
            Spectrogram.wins(
                window_size,
                seconds * self.sample_rate,
                step_size
            )
        )
        self.bins_per_track = Spectrogram.bins(fft_resolution)
        data_x = numpy.zeros(
            (len(indexes), self.wins_per_track, self.bins_per_track),
            dtype=numpy.float32)
        data_y = numpy.zeros(
            (len(indexes), len(self.genres)),
            dtype=numpy.int8)
        files_and_genres_by_index = self.files_and_genres.items()
        for data_i, index in enumerate(indexes):
            filename = files_and_genres_by_index[index][0]
            genre = files_and_genres_by_index[index][1]
            print("calculating spectrogram of " + filename)
            data_x[data_i] = Spectrogram.from_waveform(
                self.read_raw_audio(filename, seconds),
                window_size, step_size, window_type,
                fft_resolution).spectrogram
            data_y[data_i][self.genres.index(genre)] = 1
        data_x = numpy.reshape(
            data_x,
            (data_x.shape[0], data_x.shape[1] * data_x.shape[2])
        )
        return data_x, data_y

    def filter_indexes(self, indexes, data_x, data_y):
        print("filtering indexes...")
        return numpy.take(data_x, indexes, axis=0), \
            numpy.take(data_y, indexes, axis=0)

    def scale(self, data_x):
        print("preprocessing...")
        preprocessor = StandardScaler(copy=False)
        preprocessor.fit_transform(data_x)

    # @abstractmethod
    def list_audio_files_and_genres(self, audio_folder):
        """
        Returns an OrderedDict of { filename: genre }.
        Assumes that audio_folder contains genre-named folders, which
        contain audio files in that genre.
        """
        extensions = available_file_formats()
        files_and_genres = OrderedDict()
        audio_folder = os.path.expanduser(audio_folder)
        for root, dirnames, filenames in os.walk(audio_folder):
            for f in filenames:
                for x in extensions:
                    if f.endswith(x):
                        filename = os.path.join(root, f)
                        genre = os.path.basename(root)
                        files_and_genres[filename] = genre
        return files_and_genres

    def read_raw_audio(self, file, seconds):
        print("reading " + file)
        sndfile = Sndfile(file, mode='r')
        return sndfile.read_frames(
            seconds * self.sample_rate,
            dtype=numpy.float32)

    def read_all_raw_audio(self):
        """
        Returns an array of songs represented in raw audio data.
        """
        raw_audio_data = []
        for file in self.files_and_genres.iterkeys():
            raw_audio_data.append(self.read_raw_audio(file))
        return raw_audio_data

    def __get_time_dimension(self, signal_length):
        return numpy.linspace(
            0, signal_length/self.sample_rate, num=signal_length)

    def plot_raw_audio(self, raw_audio, title="Audio signal"):
        import matplotlib.pyplot as plt
        plt.title(title)
        plt.plot(self.__get_time_dimension(len(raw_audio)), raw_audio)
        plt.show()

    def raws_to_spectrograms(self, raw_audio_data, window_size, step_size,
                             window_type, fft_resolution):
        """
        Returns an array of spectrograms from the raw audio data.

        Parameters
        ----------

        raw_audio_data: an array of raw audio data read using libsndfile
        window_size: the window size for the spectrogram calculation
        step_size: the step size for the spectrogram calculation
        window_type: the window type for the spectrogram calculation
        fft_resolution: the resolution of the Fourier Transform
        """
        spectrogram_data = []
        for raw_audio in raw_audio_data:
            spectrogram_data.append(
                Spectrogram.from_waveform(
                    raw_audio, window_size, step_size, window_type,
                    fft_resolution))
        return spectrogram_data

    def plot_spectrogram(self, spectrogram, title="Spectrogram"):
        spectrogram.plot(title)

    def get_indexes(self, set, nfolds=4, run_n=0, seed=1234):
        tracks = numpy.arange(self.number_of_tracks)
        if set == 'all':
            return tracks
        else:
            rng = make_np_rng(None, seed, which_method="shuffle")
            rng.shuffle(tracks)
            kf = KFold(tracks, n_folds=nfolds)
            return kf.runs[run_n][set]
