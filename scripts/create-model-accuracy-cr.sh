#!/usr/bin/env bash

scripts/model-accuracy-cr.sh outputs/metrics | oc create -f -
