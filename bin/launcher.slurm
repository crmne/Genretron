#!/usr/bin/env bash

function require_vars {
    for var in "$@"; do
        if [ -v "$var" ]; then
            eval value=\$$var
            echo "$var=$value"
        else 
            echo "var $var was not found"
            exit 1
        fi
    done
}

. ~/.bashrc

export CONSECUTIVE_JOBS=$1
export DB_TABLE=$2

. $GENRETRON_PATH/env/bin/activate

COMMAND="$GENRETRON_PATH/bin/jobman sql -n $CONSECUTIVE_JOBS $DB_TABLE $GENRETRON_PATH/results"

echo $COMMAND
srun $COMMAND
