#!/usr/bin/env bash

cd build
rm -r *
cd ..
cmake -DCMAKE_PREFIX_PATH=../../../lib/libtorch -H. -Bbuild
cd build
make disaster-tweets
