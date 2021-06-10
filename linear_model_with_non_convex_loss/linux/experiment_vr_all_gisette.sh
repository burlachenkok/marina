#!/usr/bin/env bash

cd ./../
echo "Git revision: "
git rev-parse HEAD

export test_name=gisette_scale
mpiexec -n 6 python3.8 ./experiment_vr_all.py
