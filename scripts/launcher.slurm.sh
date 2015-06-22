#!/usr/bin/env bash
. ~/.bashrc

export CONSECUTIVE_JOBS=$1
export DB_TABLE=$2


if [ -z "$GENRETRON_PATH" ] || [ -z "$DB_TABLE" ] || [ -z "$CONSECUTIVE_JOBS" ]
    echo "One of the required variables by this script has not been set. Exiting"
    then exit 1
fi

. $GENRETRON_PATH/env/bin/activate

COMMAND="$GENRETRON_PATH/env/bin/jobman sql -n $CONSECUTIVE_JOBS $DB_TABLE $GENRETRON_PATH/results"

echo $COMMAND
srun $COMMAND
