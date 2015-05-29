#!/usr/bin/env bash
. ~/.bashrc
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
export DB_TABLE=$1

$GENRETRON_PATH/env/bin/jobman sql -n 1 $DB_TABLE $GENRETRON_PATH/results
