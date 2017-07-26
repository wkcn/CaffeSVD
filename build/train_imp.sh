#!/usr/bin/env sh
set -e

TOOLS=/home/luwei1/caffe/build/tools

$TOOLS/caffe train \
  --solver=./build/solver.prototxt \
  --weights ./build/ip2_SVD4.caffemodel $@
