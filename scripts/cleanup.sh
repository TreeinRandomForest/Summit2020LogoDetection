#!/usr/bin/env bash

for type in pipelinerun taskrun resource; do
  for p in $(tkn ${type} list | awk '{print $1}' | tail -n +2); do
    tkn ${type} delete -f ${p}
  done
done

oc -n policy-pipeline apply -f ./policy/modelaccuracythreshold.yaml
