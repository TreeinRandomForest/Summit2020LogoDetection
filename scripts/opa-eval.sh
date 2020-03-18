#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE}")/util.sh"
QUERY_STRING="data.main.deny[message]"

podman run --rm -v ${ROOT_DIR}:/project:Z -w /project openpolicyagent/opa:0.17.3 eval \
  -f pretty --fail-defined -i "${MODEL_ACCURACY_METRICS}" -d "${MODEL_ACCURACY_POLICY}" "${QUERY_STRING}"
