#!/usr/bin/env sh
set -e

TOOLS=/home/luwei1/caffe/build/tools

$TOOLS/caffe train \
  --solver=$proto \
  --weights $model $@
