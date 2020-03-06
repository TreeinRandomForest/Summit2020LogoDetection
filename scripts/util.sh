#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "$0")/.." ; pwd)"
MODEL_ACCURACY_METRICS="${MODEL_ACCURACY_METRICS:-outputs/metrics}"
MODEL_ACCURACY_POLICY="${MODEL_ACCURACY_POLICY:-policy/accuracy.rego}"

print-error-and-exit() {
  echo "$0: ERROR: ${1} not found."
  exit 1
}

if [[ ! -f ${ROOT_DIR}/${MODEL_ACCURACY_METRICS} ]]; then
  print-error-and-exit "MODEL_ACCURACY_METRICS=${ROOT_DIR}/${MODEL_ACCURACY_METRICS}"
elif [[ ! -f ${ROOT_DIR}/${MODEL_ACCURACY_POLICY} ]]; then
  print-error-and-exit "MODEL_ACCURACY_POLICY=${ROOT_DIR}/${MODEL_ACCURACY_POLICY}"
fi

