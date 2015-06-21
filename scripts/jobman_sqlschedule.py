#!/usr/bin/env python
import argparse
import jobman
import itertools
from pylearn2.scripts.jobman.experiment import ydict, train_experiment

hp = {
    'learning_rate': .001,
    'batch_size': [10, 100],
    'h0_output_channels': [16, 32, 64],
    'h0_kernel_shape_x': 513,
    'h0_kernel_shape_y': [4, 8, 16, 32],
    'h0_pool_shape_x': 1,
    'h0_pool_shape_y': 4,
    'h0_pool_stride_x': 1,
    'h0_pool_stride_y': 2,
    'h1_kernel_shape_x': 1,
    'h1_kernel_shape_y': [4, 8, 16, 32],
    'h1_pool_shape_x': 1,
    'h1_pool_shape_y': 4,
    'h1_pool_stride_x': 1,
    'h1_pool_stride_y': 2,
    'h1_output_channels': [32, 64],
    'h2_kernel_shape_x': 1,
    'h2_kernel_shape_y': [4, 8, 16, 32],
    'h2_pool_shape_x': 1,
    'h2_pool_shape_y': 4,
    'h2_pool_stride_x': 1,
    'h2_pool_stride_y': 2,
    'h2_output_channels': 64
}


def generate_hyperparameters(hyperparameters):
    dd = jobman.DD()
    fixed_hyperparameters = \
        [(k, v) for k, v in hyperparameters.items() if not isinstance(v, list)]

    dd.update(fixed_hyperparameters)

    product_hyperparameters = \
        [(k, v) for k, v in hyperparameters.items() if isinstance(v, list)]

    for p in itertools.product(*[i[1] for i in product_hyperparameters]):
        dd.update(zip([i[0] for i in product_hyperparameters], p))
        yield dd


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
    for hyperparameters in generate_hyperparameters(hp):
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
