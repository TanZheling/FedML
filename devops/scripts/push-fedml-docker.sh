#!/bin/bash

push_arm_arch_images=$1

export FEDML_VERSION=`cat python/setup.py |grep version= |awk -F'=' '{print $2}' |awk -F',' '{print $1}'|awk -F'"' '{print $2}'`
if [[ $push_arm_arch_images == "" ]]; then
  docker push fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel
  docker push fedml/fedml:${FEDML_VERSION}-torch1.13.1-cuda11.6-cudnn8-devel
fi

if [[ $push_arm_arch_images != "" ]]; then
  docker push fedml/fedml:latest-arm64-torch1.13.1-cuda11.6-cudnn8-devel
  docker push fedml/fedml:${FEDML_VERSION}-arm64-torch1.13.1-cuda11.6-cudnn8-devel-arm64

  docker push fedml/fedml:latest-nvidia-jetson-l4t-ml-r35.1.0-py3
  docker push fedml/fedml:${FEDML_VERSION}-nvidia-jetson-l4t-ml-r35.1.0-py3

#  docker push fedml/fedml:latest-raspberrypi4-32-py38
#  docker push fedml/fedml:${FEDML_VERSION}-raspberrypi4-32-py38

  docker push fedml/fedml:latest-raspberrypi4-64-py38
  docker push fedml/fedml:${FEDML_VERSION}-raspberrypi4-64-py38
fi