#!/usr/bin/env python
import argparse
import jobman
from pylearn2.scripts.jobman.experiment import ydict, train_experiment
from spectrogram import Spectrogram


# TODO: make this function load from a conf file
def generate_hyperparameters():
    hyperparameters = jobman.DD()
    hyperparameters.seed = 1234
    hyperparameters.fft_resolution = 1024
    hyperparameters.h0_irange = .05
    hyperparameters.h0_pool_shape_x = 1
    hyperparameters.h0_pool_shape_y = 2
    hyperparameters.h0_pool_stride_x = 1
    hyperparameters.h0_pool_stride_y = 1
    hyperparameters.h0_max_kernel_norm = 1.9365
    hyperparameters.h1_irange = .05
    hyperparameters.h1_kernel_shape_x = 1
    hyperparameters.h1_kernel_shape_y = 10
    hyperparameters.h1_pool_shape_x = 1
    hyperparameters.h1_pool_shape_y = 2
    hyperparameters.h1_pool_stride_x = 1
    hyperparameters.h1_pool_stride_y = 1
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

                        for h0_kernel_shape_y in hyperparameters.input_shape_y / 8, \
                                hyperparameters.input_shape_y / 16:
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
        state.hyper_parameters = \
            jobman.expand(jobman.flatten(hyperparameters), dict_type=ydict)
        state.extract_results = "jobman_utils.results_extractor"
        jobman.sql.insert_job(
            train_experiment,
            jobman.flatten(state),
            db)

    # create view
    view_db_path = "{0}_view".format(args.table_name)
    db.createView(view_db_path)
