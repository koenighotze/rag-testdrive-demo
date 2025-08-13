#!/usr/bin/env bash

# when a command fails, bash exits instead of continuing with the rest of the script
set -o errexit
# make the script fail, when accessing an unset variable
set -o nounset
# pipeline command is treated as failed, even if one command in the pipeline fails
set -o pipefail
# enable debug mode, by running your script as TRACE=1
if [[ "${TRACE-0}" == "1" ]]; then set -o xtrace; fi

docker run \
	-p 6333:6333 \
	-p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
	--name "qdrant" \
    qdrant/qdrant:dev
