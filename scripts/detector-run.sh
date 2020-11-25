#!/bin/bash

echo 'Building the detector docker container...'

echo 'Sync-ing s3 so that we have *.pth files available...'
./scripts/sync-s3.sh

echo 'Building container (this will take > an hour the first time, as it builds OpenCV2 with CUDA+CUDNN support...'
docker build . -f src/docker/Dockerfile.rate_detector -t detector

echo 'Running container...'
docker run --net=host --runtime nvidia --rm --ipc=host \
  -v /tmp/.X11-unix/:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY \
  --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 \
  -v $(pwd):/repo -it detector
