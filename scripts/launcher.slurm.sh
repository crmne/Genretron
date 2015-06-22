#!/usr/bin/env bash

function require_vars {
    for var in "$@"
    do
        if [ -z "$var" ]
            echo 
            then exit 1
        fi
}

. ~/.bashrc

export CONSECUTIVE_JOBS=$1
export DB_TABLE=$2

. $GENRETRON_PATH/env/bin/activate

COMMAND="$GENRETRON_PATH/env/bin/jobman sql -n $CONSECUTIVE_JOBS $DB_TABLE $GENRETRON_PATH/results"

echo $COMMAND
srun $COMMAND
