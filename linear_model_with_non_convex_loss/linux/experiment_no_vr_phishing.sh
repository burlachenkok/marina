#!/usr/bin/env bash

cd ./../
echo "Git revision: "
git rev-parse HEAD

export test_name=phishing
mpiexec -n 6 python3.8 ./experiment_no_vr_all.py
