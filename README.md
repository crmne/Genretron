# Genretron

Genretron aims to be an automatic music genre classifier. That means that given any song (even one that you just made) it will try to classify it in one of its learned genres. It uses Deep Learning techniques to learn the genre of the songs.

Genretron is part of Carmine Paolino's Master Thesis at Universiteit van Amsterdam.

## Installation

### Dependencies

Install Python 2.7+ with virtualenv, HDF5, libsndfile, freetype, libpng, a fortran compiler and postgresql.

On a Mac with [Homebrew](http://brew.sh):

    brew install python libsndfile homebrew/science/hdf5 freetype libpng gcc postgresql homebrew/python/matplotlib homebrew/python/numpy homebrew/python/scipy
    pip install virtualenv

On Ubuntu 14.04 LTS:

    sudo apt-get install libsndfile1-dev python-virtualenv python-tables python-matplotlib python-scikits-learn python-sqlalchemy python-psycopg2 python-yaml python-numpy python-scipy python-progressbar procmail

### Genretron

    git clone --recursive https://github.com/crmne/Genretron.git
    cd Genretron
    virtualenv --system-site-packages env
    . env/bin/activate
    python setup.py install

## Downloading the datasets

Make sure you set the environment variable `PYLEARN2_DATA_PATH` to a directory of your preference where you want the datasets to be stored. For example:

    export PYLEARN2_DATA_PATH=/data/lisa/data

For your own convenience, you might want to add that line to your .bashrc

Then simply run

    bin/download_gtzan

This procedure will download the classic Music Information Retrieval Genre Classification dataset GTZAN and unpack it in your `PYLEARN2_DATA_PATH` directory so it's ready for use with Genretron.

If you wanna download MNIST, run:

    python bin/download_mnist

## Training of the neural network

    bin/train experiments/test/conv.yaml

### Training on a cluster with Jobman

Set the `GENRETRON_PATH`, `DB_TABLE` and `THEANO_FLAGS` environment variables. For example:

    export GENRETRON_PATH=$HOME/Genretron
    export DB_TABLE='postgres://user[:pass]@host/dbname?table=tablename'
    export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

Then schedule the jobs with jobman:

    bin/jobman_sqlschedule experiments/jobman/hyperparameters/experiment_name.yaml $DB_TABLE tablename

#### Execution of the training jobs on a single machine

    bin/jobman sql -n [number of jobs] $DB_TABLE $GENRETRON_PATH/resuts

#### Execution of the training jobs on a cluster with SLURM

    sbatch -t [time in minutes] -p [cluster partition] -n [number of processes] -N [number of nodes] bin/launcher.slurm 1 $DB_TABLE

#### Execution of the training jobs on a cluster with PBS

    for i in {1..[number of jobs]}; do qsub -lwalltime=[time in hh:mm:ss] -v CONSECUTIVE_JOBS=1,DB_TABLE=$DB_TABLE bin/launcher.pbs; done

## Prediction

    bin/predict results/experiments/test/conv.pkl song1.wav song2.wav ...
