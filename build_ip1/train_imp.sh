#!/usr/bin/env sh
set -e

TOOLS=/home/luwei1/caffe/build/tools

$TOOLS/caffe train \
  --solver=./build_ip1/solver.prototxt \
  --weights ./build_ip1/ip1_SVD6.caffemodel $@
