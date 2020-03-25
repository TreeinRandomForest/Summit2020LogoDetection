#!/usr/bin/env bash

METRICS_FILE=${1:-}

function usage() {
  echo "Usage: $0 [METRICS_FILE]"
}

function validate-args() {
  if [[ -z ${METRICS_FILE} ]]; then
    echo "ERROR: missing required argument [METRICS_FILE]"
    usage
    exit 1
  elif [[ ! -f ${METRICS_FILE} ]]; then
    echo "ERROR: ${METRICS_FILE} metrics file not found..."
    usage
    exit 1
  fi
}

function get-accuracy() {
  cat ${1} | awk -F": " '{print $2}' | awk -F"}" '{print $1}'
}

function convert-to-percent() {
  awk "BEGIN {print $1 * 100} "
}

function output-cr-yaml() {
  cat <<EOF
apiVersion: ai.rhsummit2020.cloud/v1alpha1
kind: ModelAccuracy
metadata:
  generateName: modelaccuracy-
spec:
  accuracy: ${1}
EOF
}

function main() {
  validate-args $*
  #accuracy="$(cat ${METRICS_FILE} | awk -F": " '{print $2}' | awk -F"}" '{print $1}')"
  local accuracy="$(get-accuracy ${METRICS_FILE})"
  accuracy="$(convert-to-percent ${accuracy})"
  output-cr-yaml ${accuracy}
}

main
