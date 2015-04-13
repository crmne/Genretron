Genretron-Pylearn2
==================

Installation
------------

Make sure you have python 2.7+ installed and then:

	git submodule update --init
	virtualenv env
	. env/bin/activate
    pip install numexpr numpy cython
    pip install -r requirements.txt
    echo "`pwd`" > env/lib/python2.7/site-packages/genretron.pth
    echo "`pwd`/pylearn2" > env/lib/python2.7/site-packages/pylearn2.pth
    . env/bin/activate

Training
--------

	. env/bin/activate
    pylearn2/pylearn2/scripts/train.py conv.yaml
