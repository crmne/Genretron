# --
# Launcher for PBS job scheduling systems.
# To launch a job with this script:
# $ qsub -v GENRETRON_PATH="$HOME/Genretron-Pylearn2",EXPERIMENT="single_experiments/mlp_20150107A.yaml" launcher.pbs.sh
# --
# shell for the job:
#PBS -S /bin/bash
# job requires at most 3 hours, 0 minutes
#     and 0 seconds wallclock time and uses one 16-core node:
#PBS -lwalltime=3:00:00 -lnodes=1:cores16
# cd to the directory where the program is to be called:
if [ "x" == "x$GENRETRON_PATH" ]
then echo "GENRETRON_PATH variable not set!! Quitting!"
else cd $GENRETRON_PATH
fi
# load the Intel Compilers and Math Kernel Library
module load fortran mkl
module load fortran/intel
module load c/intel
# request that threads should remain on the same core:
module load paffinity
# activate the python virtualenv
. env/bin/activate
# launch the learning procedure
if [ "x" == "x$EXPERIMENT" ]
then echo "EXPERIMENT variable not set!! Quitting!"
else pylearn2/pylearn2/scripts/train.py $EXPERIMENT
fi
