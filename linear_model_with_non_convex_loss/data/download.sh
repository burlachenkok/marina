#!/usr/bin/env bash
function load 
{
  # extract last name in path separated by backslashes
  url=${1}
  fname=${url##*/}

  curl "${url}" --output ${fname} --fail --silent
  if ((${?}==0)); then
    echo "[OK] file \"${fname}\" has been downloaded successfully (train)"
  else
    echo "[ERROR] file \"${fname}\" can not be downloaded (train)"
  fi

  # extract last name in path separated by backslashes
  url=${1}.t
  fname=${url##*/}

  curl "${url}" --output ${fname} --fail --silent
  if ((${?}==0)); then
    echo "[OK] file \"${fname}\" has been downloaded successfully (test)"
  else
    echo "[ERROR] file \"${fname}\" can not be downloaded (test)"
  fi
}

function load_train_only
{
  # extract last name in path separated by backslashes
  url=${1}
  fname=${url##*/}

  curl "${url}" --output ${fname} --fail --silent
  if ((${?}==0)); then
    echo "[OK] file \"${fname}\" has been downloaded successfully (train)"
  else
    echo "[ERROR] file \"${fname}\" can not be downloaded (train)"
  fi
}


load "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
load "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon"
load "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"

load_train_only "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2"
load_train_only "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing"
load_train_only "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/duke.bz2"
