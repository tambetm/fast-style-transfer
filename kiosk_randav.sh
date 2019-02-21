#!/bin/bash

conda activate faststyle

cd /home/arvuti/fast-style-transfer
while true; do
    python eval_camera.py --detect_faces --stylize_preview --capture_device 0 --vertical --fullscreen $*
done
