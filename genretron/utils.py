import sys

__authors__ = "Carmine Paolino"
__copyright__ = "Copyright 2015, Vrije Universiteit Amsterdam"
__credits__ = ["Carmine Paolino"]
__license__ = "3-clause BSD"
__email__ = "carmine@paolino.me"


def filter_keys_from_dict(keys, dict):
    key_set = set(dict) - set(list(keys))
    return {k: dict[k] for k in key_set}


def filter_null_args(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


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


def longest_common_substring(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and __is_substr(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr


def __is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True


def urlretrieve(url, filename):
    from progressbar import Percentage, Bar, ETA, FileTransferSpeed, ProgressBar
    import urllib
    import os
    widgets = [os.path.basename(filename), ' ', Percentage(), ' ',
               Bar(), ' ', ETA(), ' ',
               FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)

    def dlProgress(count, blockSize, totalSize):
        if pbar.maxval is None:
            pbar.maxval = totalSize
            pbar.start()

        pbar.update(min(count*blockSize, totalSize))

    print("Downloading {0}".format(url))
    urllib.urlretrieve(url, filename, reporthook=dlProgress)
    pbar.finish()
