#!/bin/sh

python eval_camera.py --capture_device 0 --detect_faces --stylize_preview --vertical --fullscreen $*
