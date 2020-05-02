#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE}")/util.sh"

podman run --rm -v "${ROOT_DIR}":/project:Z instrumenta/conftest:v0.17.0 test \
  -i json "${MODEL_ACCURACY_METRICS}" -p "${MODEL_ACCURACY_POLICY}"
