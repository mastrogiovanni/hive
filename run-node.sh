#!/bin/bash

docker run \
    -v /home/michele/.cache/huggingface:/root/.cache/huggingface \
    --gpus all \
    mastrogiovanni/hive-node:v0.0.1 \
    --control-socket 192.168.1.102:9000 \
    --parts 3 \
    --index 0 \
    --model "Qwen/Qwen3-0.6B"
