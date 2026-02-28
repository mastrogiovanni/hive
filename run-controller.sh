#!/bin/bash

docker run \
    -e PYTHONUNBUFFERED=1 \
    -v /home/michele/.cache/huggingface:/root/.cache/huggingface \
    --gpus all \
    -p 9000:9000 \
    mastrogiovanni/hive-controller run control.py --bind "0.0.0.0:9000"
