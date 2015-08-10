import numpy


class ZNormalizer(object):
    def fit_transform(self, data):
        mean = numpy.mean(data)
        std = numpy.std(data)
        data -= mean
        data /= std
        return data


class LinearNormalizer(object):
    def fit_transform(self, data):
        raise NotImplementedError


class OutlierReplacer(object):
    def fit_transform(self, data, percentile=5):
        raise NotImplementedError


preprocessors = {
    'znormalizer': ZNormalizer(),
    'linearnormalizer': LinearNormalizer(),
    'outlierreplacer': OutlierReplacer()
}


def preprocessor_factory(preprocessor):
    return preprocessors[preprocessor]

