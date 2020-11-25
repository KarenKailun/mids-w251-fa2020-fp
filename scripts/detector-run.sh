#!/bin/bash
docker build . -f src/docker/Dockerfile.rate_detector -t detector
docker run --net=host --runtime nvidia --rm --ipc=host \
  -v /tmp/.X11-unix/:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY \
  --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 \
  -v $(pwd):/repo -it detector
