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
echo "Job $PBS_JOBID started at `date`" | mail $USER -s "Job $PBS_JOBID"
# load the Intel Compilers and Math Kernel Library
module load fortran mkl
module load fortran/intel
module load c/intel
# request that threads should remain on the same core:
module load paffinity
# check if environment variables are correctly set
if [ "x" == "x$GENRETRON_PATH" || "x" == "x$EXPERIMENT" || "x" == "x$EXPERIMENTS_PATH" ]
then echo "One or more variable are not set!! Quitting!"
exit 1
fi
# activate the python virtualenv
. "${GENRETRON_PATH}"/env/bin/activate
# launch the learning procedure
"${GENRETRON_PATH}"/pylearn2/pylearn2/scripts/train.py "$EXPERIMENT"
# move the results of the experiment in the proper folder
exp_dirname=$(dirname "$EXPERIMENT")
exp_filename=$(basename "$EXPERIMENT")
exp_name="${exp_filename%.*}"
mv "${exp_dirname}/${exp_name}.pkl" "$EXPERIMENTS_PATH"
mv "${exp_dirname}/${exp_name}_best.pkl" "$EXPERIMENTS_PATH"
echo "Job $PBS_JOBID completed at `date`" | mail $USER -s "Job $PBS_JOBID"
