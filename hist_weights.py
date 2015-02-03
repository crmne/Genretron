#!/usr/bin/env python
"""
Shows a histogram of the weights of the model.
"""
import argparse
import pylab
import cPickle as pickle


def show_weights(model_path):
    """
    Shows a histogram of the weights of the model.

    Parameters
    ----------
    model_path : str
        Path of the model to show weights for
    """
    pkl = pickle.load(open(model_path, 'rb'))
    w = pkl.get_weights_topo().flatten()
    n, bins, patches = pylab.hist(w, 50)
    pylab.show()


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    show_weights(args.path)
