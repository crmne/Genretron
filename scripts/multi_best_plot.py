#!/usr/bin/env python
"""
Plots the best training, test and validation values of all the models passed.
"""
import os
import argparse
import numpy
import pylab
import cPickle as pickle
import pylearn2


def extract_channels(model_path, channels):
    print("Extracting monitoring channels from %s" %
          os.path.basename(model_path))
    pkl = pickle.load(open(model_path, 'rb'))
    chans = {
        channel: pkl.monitor.channels[channel].val_record[-1]
        for channel in channels
    }
    return chans


def plot(models, channels, param=None):
    model_names = []
    m_vals = []
    for model in models:
        model_names.append(os.path.basename(model).replace("_best.pkl", ''))
        m_vals.append(extract_channels(model, channels))
    x = numpy.arange(len(model_names))

    for channel in channels:
        y = numpy.asarray([m[channel] for m in m_vals])
        pylab.plot(x, y, label=channel, marker='.')

    pylab.legend(channels, loc='upper center')

    if param:
        params = []
        for model_name in model_names:
            model = pylearn2.config.yaml_parse.load(
                open(model_name + '.yaml'), instantiate=False)
            # FIXME
            # extracting l1 regularization coefficient:
            # generalize to anything in the yaml file
            params.append(
                model[2]['algorithm'][2]['cost'][2]
                ['costs'][1][2]['coeffs']['h2']
            )
        x = numpy.asarray(params)
        pylab.xlabel(param)
    else:
        pylab.xticks(x, model_names, rotation=70)

    pylab.show()


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", nargs='+')
    parser.add_argument("--channels", nargs='+')
    parser.add_argument("--param")
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    plot(args.model_paths, args.channels, args.param)
