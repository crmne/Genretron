#!/usr/bin/env bash
. ~/.bashrc

if [ -z "$GENRETRON_PATH" ] || [ -z "$DB_TABLE" ] || [ -z "$1" ]
    echo "One of the required variables by this script has not been set. Exiting"
    then exit 1
fi

. $GENRETRON_PATH/env/bin/activate
srun $GENRETRON_PATH/env/bin/jobman sql -n $1 $DB_TABLE $GENRETRON_PATH/results
