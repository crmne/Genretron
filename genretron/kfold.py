import numpy

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


class KFold(object):
    """KFold CrossValidation with support for validation sets"""
    def __init__(self, idxs, n_folds=4):
        assert n_folds >= 3
        assert isinstance(idxs, numpy.ndarray)
        self.n_folds = n_folds
        self.runs = []
        folds = numpy.split(idxs, self.n_folds)
        for run_n in xrange(self.n_folds):
            run = {'train': [], 'valid': [], 'test': []}
            test_idxs = (0 + run_n) % self.n_folds
            valid_idxs = (1 + run_n) % self.n_folds
            run['test'] = folds[test_idxs]
            run['valid'] = folds[valid_idxs]
            run['train'] = numpy.concatenate(
                [x for i, x in enumerate(folds) if i not in {test_idxs, valid_idxs}])
            self.runs.append(run)

