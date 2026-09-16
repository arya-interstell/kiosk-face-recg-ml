#!/usr/bin/env sh
# YuNet face detector from the official OpenCV model zoo (BSD-3-Clause).
set -e
cd "$(dirname "$0")"
curl -sSL -O https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
echo "fetched face_detection_yunet_2023mar.onnx"
curl -sSL -O https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
echo "fetched face_recognition_sface_2021dec.onnx"
