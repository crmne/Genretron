#!/usr/bin/env bash
#SBATCH -p gpu_short
#SBATCH -t 30
#SBATCH -N 5
. ~/.bashrc
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
export DB_TABLE=$1

srun $GENRETRON_PATH/env/bin/jobman sql -n 1 $DB_TABLE $GENRETRON_PATH/results
