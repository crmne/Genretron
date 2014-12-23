import numpy
import sys
from jobman.tools import DD


def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    valid_y_misclass = channels['valid_y_misclass'].val_record[-1]

    return DD(valid_y_misclass=valid_y_misclass)


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


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
