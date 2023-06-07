#!/usr/bin/env bash

cmake -DCMAKE_PREFIX_PATH=../../../lib/libtorch -H. -Bbuild
cd build
make disaster-tweets
./disaster-tweets ../data/train.csv ../data/glove.txt
