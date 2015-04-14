Genretron-Pylearn2
==================

Genretron-Pylearn2 aims to be an automatic music genre classifier. 

Installation
------------

Make sure you have python 2.7+ installed and then:

	git submodule update --init
	virtualenv env
	. env/bin/activate
    pip install -r requirements.txt
    echo "`pwd`/src" > env/lib/python2.7/site-packages/genretron.pth
    echo "`pwd`/pylearn2" > env/lib/python2.7/site-packages/pylearn2.pth
    . env/bin/activate

Training of the neural network
------------------------------

	. env/bin/activate
    scripts/train.py experiments/test/conv.yaml

Prediction of the genre of a song
---------------------------------

