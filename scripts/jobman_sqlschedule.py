#!/usr/bin/env python
import argparse
import jobman
import jobman_utils
import pylearn2.config.yaml_parse
from theano.compat import six
from spectrogram import Spectrogram


# code from pylearn2
class ydict(dict):
    '''
    YAML-friendly subclass of dictionary.

    The special key "__builder__" is interpreted as the name of an object
    constructor.

    For instance, building a ydict from the following dictionary:

        {
            '__builder__': 'pylearn2.training_algorithms.sgd.EpochCounter',
            'max_epochs': 2
        }

    Will be displayed like:

        !obj:pylearn2.training_algorithms.sgd.EpochCounter {'max_epochs': 2}
    '''
    def __str__(self):
        args_dict = dict(self)
        builder = args_dict.pop('__builder__', '')
        ret_list = []
        if builder:
            ret_list.append('!obj:%s {' % builder)
        else:
            ret_list.append('{')

        for key, val in six.iteritems(args_dict):
            # This will call str() on keys and values, not repr(), so unicode
            # objects will have the form 'blah', not "u'blah'".
            ret_list.append('%s: %s,' % (key, val))

        ret_list.append('}')
        return '\n'.join(ret_list)


# code from pylearn2
def train_experiment(state, channel):
    """
    Train a model specified in state, and extract required results.

    This function builds a YAML string from ``state.yaml_template``, taking
    the values of hyper-parameters from ``state.hyper_parameters``, creates
    the corresponding object and trains it (like train.py), then run the
    function in ``state.extract_results`` on it, and store the returned values
    into ``state.results``.

    To know how to use this function, you can check the example in tester.py
    (in the same directory).
    """
    yaml_template = state.yaml_template

    # Convert nested DD into nested ydict.
    hyper_parameters = \
        jobman.expand(jobman.flatten(state.hyper_parameters), dict_type=ydict)

    # This will be the complete yaml string that should be executed
    final_yaml_str = yaml_template % hyper_parameters

    # Instantiate an object from YAML string
    train_obj = pylearn2.config.yaml_parse.load(final_yaml_str)

    try:
        iter(train_obj)
        iterable = True
    except TypeError:
        iterable = False
    if iterable:
        raise NotImplementedError(
                ('Current implementation does not support running multiple '
                 'models in one yaml string.  Please change the yaml template '
                 'and parameters to contain only one single model.'))
    else:
        # print "Executing the model."
        train_obj.main_loop()
        # This line will call a function defined by the user and pass train_obj
        # to it.
        state.results = jobman.tools.resolve(state.extract_results)(train_obj)
        return channel.COMPLETE


# TODO: make this function load from a conf file
def generate_hyperparameters():
    hyperparameters = jobman.DD()
    hyperparameters.seed = 1234
    hyperparameters.fft_resolution = 1024
    hyperparameters.h0_irange = .05
    hyperparameters.h0_pool_shape_x = 1
    hyperparameters.h0_pool_shape_y = 8
    hyperparameters.h0_pool_stride_x = 1
    hyperparameters.h0_pool_stride_y = 4
    hyperparameters.h0_max_kernel_norm = 1.9365
    hyperparameters.h1_irange = .05
    hyperparameters.h1_kernel_shape_x = 1
    hyperparameters.h1_kernel_shape_y = 10
    hyperparameters.h1_pool_shape_x = 1
    hyperparameters.h1_pool_shape_y = 8
    hyperparameters.h1_pool_stride_x = 1
    hyperparameters.h1_pool_stride_y = 4
    hyperparameters.h1_output_channels = 64
    hyperparameters.h1_max_kernel_norm = 1.9365
    hyperparameters.y_max_col_norm = 1.9365
    hyperparameters.y_istdev = .05
    hyperparameters.learning_rate = .001
    hyperparameters.prop_decrease = 0.001
    hyperparameters.prop_decrease_n = 2
    for window_size in 1024, 2048:
        hyperparameters.window_size = window_size
        hyperparameters.input_shape_x = \
            Spectrogram.bins(hyperparameters.fft_resolution)

        for seconds in 1., 4., 29.:
            hyperparameters.seconds = seconds
            hyperparameters.input_shape_y = \
                len(Spectrogram.wins(
                    seconds * 22050,
                    window_size,
                    window_size / 2))

            for batch_size in 1, 10, 100:
                hyperparameters.batch_size = batch_size

                for h0_output_channels in 1, 16, 32, 64, 128:
                    hyperparameters.h0_output_channels = h0_output_channels

                    for h0_kernel_shape_x in hyperparameters.input_shape_x, \
                            hyperparameters.input_shape_x / 4:
                        hyperparameters.h0_kernel_shape_x = h0_kernel_shape_x

                        for h0_kernel_shape_y in hyperparameters.input_shape_y, \
                            hyperparameters.input_shape_y / 2, \
                            hyperparameters.input_shape_y / 3, \
                                hyperparameters.input_shape_y / 4:
                            hyperparameters.h0_kernel_shape_y = h0_kernel_shape_y
                            yield hyperparameters


def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path')
    parser.add_argument('db_path')
    parser.add_argument('table_name')
    return parser

if __name__ == '__main__':
    # parse arguments
    parser = make_argument_parser()
    args = parser.parse_args()

    # open connection to db
    full_db_path = "{0}?table={1}".format(args.db_path, args.table_name)
    db = jobman.api0.open_db(full_db_path)

    # read yaml template
    with open(args.yaml_path) as f:
        yaml_template = f.read()

    # generate experiments and put them in db
    for hyperparameters in generate_hyperparameters():
        state = jobman.DD()
        state.yaml_template = yaml_template
        state.hyperparameters = \
            jobman.expand(jobman.flatten(hyperparameters), dict_type=ydict)
        state.extract_results = "jobman_utils.results_extractor"
        jobman.sql.insert_job(
            train_experiment,
            jobman.flatten(state),
            db)

    # create view
    view_db_path = "{0}_view".format(args.table_name)
    db.createView(view_db_path)
