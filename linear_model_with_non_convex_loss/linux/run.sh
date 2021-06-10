#!/usr/bin/env bash

export one_test=1

./experiment_no_vr_a9a.sh > experiment_no_vr_a9a.txt &
./experiment_no_vr_all_mushroom.sh > experiment_no_vr_all_mushroom.txt &
./experiment_no_vr_duke.sh > experiment_no_vr_duke.txt &
./experiment_no_vr_gisette.sh > experiment_no_vr_gisette.txt &
./experiment_no_vr_madelon.sh > experiment_no_vr_madelon.txt &
./experiment_no_vr_phishing.sh > experiment_no_vr_phishing.txt &
./experiment_no_vr_w8a.sh > experiment_no_vr_w8a.txt &
./experiment_vr_all_a9a.sh > experiment_vr_all_a9a.txt &
./experiment_vr_all_duke.sh > experiment_vr_all_duke.txt &
./experiment_vr_all_gisette.sh > experiment_vr_all_gisette.txt &
./experiment_vr_all_madelon.sh > experiment_vr_all_madelon.txt &
./experiment_vr_all_mushroom.sh > experiment_vr_all_mushroom.txt &
./experiment_vr_all_phishing.sh > experiment_vr_all_phishing.txt &
./experiment_vr_all_w8a.sh > experiment_vr_all_w8a.txt &
