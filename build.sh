#!/bin/bash
# Build from project root so COPY paths (src/, control.py, etc.) resolve.
cd "$(dirname "$0")"

docker build -f docker/Dockerfile.controller -t mastrogiovanni/hive-controller:v0.0.1 .
docker build -f docker/Dockerfile.node -t mastrogiovanni/hive-node:v0.0.1 .

