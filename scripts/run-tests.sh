#!/usr/bin/env bash

# when a command fails, bash exits instead of continuing with the rest of the script
set -o errexit
# make the script fail, when accessing an unset variable
set -o nounset
# pipeline command is treated as failed, even if one command in the pipeline fails
set -o pipefail
# enable debug mode, by running your script as TRACE=1
if [[ "${TRACE-0}" == "1" ]]; then set -o xtrace; fi

log_file="test_$$.log"

http -v :8080/query < sample_unsafe_query.json | tee -a "$log_file"
http -v :8080/query < sample_query.json | tee -a "$log_file"
