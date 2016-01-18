# -*- coding: utf-8 -*-
import numpy


class ZNormalizer(object):

    def fit_transform(self, data):
        mean = numpy.mean(data)
        std = numpy.std(data)
        data -= mean
        data /= std
        return data


class MinMaxScaler(object):

    def fit_transform(self, data, feature_range=(-1, 1)):
        _min, _max = feature_range
        std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        return std * (_max - _min) + _min


class LinearNormalizer(object):

    def fit_transform(self, data):
        raise NotImplementedError


class OutlierReplacer(object):

    def fit_transform(self, data, percentile=5):
        raise NotImplementedError


preprocessors = {
    'znormalizer': ZNormalizer(),
    'linearnormalizer': LinearNormalizer(),
    'outlierreplacer': OutlierReplacer(),
    'minmaxscaler': MinMaxScaler()
}


def preprocessor_factory(preprocessor):
    return preprocessors[preprocessor]
