#!/bin/bash

docker build -f Dockerfile.controller -t mastrogiovanni/hive-controller:v0.0.1 .

docker build -f Dockerfile.node -t mastrogiovanni/hive-node:v0.0.1 .

