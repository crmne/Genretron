Genretron-Pylearn2
==================

Installation
------------

	git submodule init
	git submodule update
	virtualenv env
	. env/bin/activate
    pip install -r requirements.txt
    echo "`pwd`" > env/lib/*/site-packages/genretron.pth
    echo "`pwd`/pylearn2" > env/lib/*/site-packages/pylearn2.pth
    . env/bin/activate

Training
--------

	. env/bin/activate
    jobman cmdline pylearn2.scripts.jobman.experiment.train_experiment mlp.conf
