# Genretron

Genretron aims to be an automatic music genre classifier. That means that given any song (even one that you just made) it will try to classify it in one of its learned genres. It uses Deep Learning techniques to learn the genre of songs.

Genretron is part of Carmine Paolino's Master Thesis at Universiteit van Amsterdam.

## Installation

Make sure you have Python 2.7+, HDF5, libsndfile, freetype, and libpng installed and run:

    git clone --recursive https://github.com/crmne/Genretron.git
    cd Genretron
    virtualenv env
    . env/bin/activate
    python setup.py install

## Downloading the dataset

Make sure you set the environment variable `PYLEARN2_DATA_PATH` to a directory of your preference where you want the datasets to be stored. For example:

    export PYLEARN2_DATA_PATH=/data/lisa/data

For your own convenience, you might want to add that line to your .bashrc

Then simply run

    bin/download_gtzan

This procedure will download the classic Music Information Retrieval Genre Classification dataset GTZAN and unpack it in your `PYLEARN2_DATA_PATH` directory so it's ready for use with Genretron.

## Training of the neural network

    bin/train experiments/test/conv.yaml

### Training on a SLURM cluster with Jobman

Set the `GENRETRON_PATH`, `DB_TABLE` and `THEANO_FLAGS` environment variables. For example:

    export GENRETRON_PATH=$HOME/Genretron
    export DB_TABLE='postgres://user[:pass]@host/dbname?table=tablename'
    export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

Then schedule the jobs with jobman:

    bin/jobman sqlschedule $DB_TABLE pylearn2.scripts.jobman.experiment.train_experiment path/to/experiment.conf

And execute them:

    sbatch -t [time in minutes] -p [cluster partition] -n [number of processes] -N [number of nodes] scripts/launcher.slurm $DB_TABLE


## Prediction

    bin/predict results/experiments/test/conv.pkl song1.wav song2.wav ...
