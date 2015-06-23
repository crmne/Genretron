#!/usr/bin/env python
import sys
import argparse
import jobman
import itertools
import yaml
from pylearn2.scripts.jobman.experiment import ydict, train_experiment


def generate_hyper_parameters(hyperparameters):
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
    parser.add_argument('yaml_config_path')
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

    # read config
    with open(args.yaml_config_path) as f:
        config = yaml.load(f.read())

    with open(config['yaml_template']) as f:
        yaml_template = f.read()

    extract_results_function = config['extract_results']
    hyper_parameters = config['hyper_parameters']

    sys.stdout.write('Generating experiments and sending them to the db')
    # generate experiments and put them in db
    for h in generate_hyper_parameters(hyper_parameters):
        state = jobman.DD()
        state.yaml_template = yaml_template
        state.hyper_parameters = \
            jobman.expand(jobman.flatten(h), dict_type=ydict)
        state.extract_results = extract_results_function
        jobman.sql.insert_job(
            train_experiment,
            jobman.flatten(state),
            db)
        sys.stdout.write('.')
        sys.stdout.flush()
    print('')

    # create view
    view_db_path = "{0}_view".format(args.table_name)
    db.createView(view_db_path)
