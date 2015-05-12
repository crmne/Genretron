Genretron-Pylearn2
==================

Genretron-Pylearn2 aims to be an automatic music genre classifier. That means that given any song (even one that you just made) it will try to classify it in one of its learned genres. It uses Deep Learning techniques to learn the genre of songs.

Genretron-Pylearn2 is part of Carmine Paolino's Master Thesis at Universiteit van Amsterdam.

Installation
------------

Make sure you have python 2.7+, HDF5, and libsndfile installed and then:

    git submodule update --init
    virtualenv env
    . env/bin/activate
    pip install -r requirements.txt
    echo $(pwd)/src > env/lib/python2.7/site-packages/genretron.pth
    echo $(pwd)/pylearn2 > env/lib/python2.7/site-packages/pylearn2.pth
    . env/bin/activate

Downloading the dataset
-----------------------

Make sure you set the environment variable `PYLEARN2_DATA_PATH` to a directory of your preference where you want the datasets to be stored. For example:

    export PYLEARN2_DATA_PATH=/data/lisa/data

For your own convenience, you might want to add that line to your .bashrc

Then simply run

    scripts/download_gtzan.py

This procedure will download the classic Music Information Retrieval Genre Classification dataset GTZAN and unpack it in your `PYLEARN2_DATA_PATH` directory so it's ready for use with Genretron.

Training of the neural network
------------------------------

	. env/bin/activate
    scripts/train.py experiments/test/conv.yaml

Prediction
----------

    . env/bin/activate
    scripts/predict.py results/experiments/test/conv.pkl song1.wav song2.wav ...
