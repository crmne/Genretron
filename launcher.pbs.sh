# --
# Launcher for PBS job scheduling systems.
# To launch a job with this script:
# $ qsub -lwalltime=3:00:0 \
#     -v GENRETRON_PATH="$HOME/Genretron-Pylearn2",\
#     EXPERIMENT="single_experiments/mlp_20150107A.yaml" \
#     -o $HOME/Experiments/Genretron-Pylearn2/mlp_20150107A.log \
#     -j oe launcher.pbs.sh
# --
# shell for the job:
#PBS -S /bin/bash
# job uses one 16-core node:
#PBS -lnodes=1:cores16
# load the Intel Compilers and Math Kernel Library
module load fortran mkl
module load fortran/intel
module load c/intel
# request that threads should remain on the same core:
module load paffinity
# check if environment variables are correctly set
if [ "x" == "x$GENRETRON_PATH" || "x" == "x$EXPERIMENT" ]
then echo "One or more variable are not set!! Quitting!"
exit 1
fi
# activate the python virtualenv
. ${GENRETRON_PATH}/env/bin/activate
# launch the learning procedure
${GENRETRON_PATH}/pylearn2/pylearn2/scripts/train.py $EXPERIMENT
