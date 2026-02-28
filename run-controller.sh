#!/bin/bash

docker run \
    -v /home/michele/.cache/huggingface:/root/.cache/huggingface \
    --gpus all \
    mastrogiovanni/hive-controller:v0.0.1 --bind "0.0.0.0:9000"
