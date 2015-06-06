import numpy
from jobman.tools import DD

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    epoch_time = channels['valid_y_misclass'].time_record
    validyma = [i.item() for i in channels['valid_y_misclass'].val_record]
    testyma = [i.item() for i in channels['test_y_misclass'].val_record]
    trainyma = [i.item() for i in channels['train_y_misclass'].val_record]
    train_time = epoch_time[-1]
    validym = validyma[-1]
    testym = testyma[-1]
    trainym = trainyma[-1]

    return DD(
        epoch_time=epoch_time,
        valid_y_misclass_array=validyma,
        test_y_misclass_array=testyma,
        train_y_misclass_array=trainyma,
        train_time=train_time,
        valid_y_misclass=validym,
        test_y_misclass=testym,
        train_y_misclass=trainym,
    )


def log_uniform(low, high):
    """
    Generates a number that's uniformly distributed in the log-space between
    `low` and `high`

    Parameters
    ----------
    low : float
        Lower bound of the randomly generated number
    high : float
        Upper bound of the randomly generated number

    Returns
    -------
    rval : float
        Random number uniformly distributed in the log-space specified by `low`
        and `high`
    """
    log_low = numpy.log(low)
    log_high = numpy.log(high)

    log_rval = numpy.random.uniform(log_low, log_high)
    rval = float(numpy.exp(log_rval))

    return rval
