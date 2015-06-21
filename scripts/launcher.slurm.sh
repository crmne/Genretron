#!/usr/bin/env bash
. ~/.bashrc

. $GENRETRON_PATH/env/bin/activate
srun $GENRETRON_PATH/env/bin/jobman sql -n $1 $DB_TABLE $GENRETRON_PATH/results
