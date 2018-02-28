#!/bin/bash

xset -dpms
xset s off
openbox-session &

xrandr --output HDMI-0 --rotate left

cd /home/tambet/fast-style-transfer
while true; do
    python eval_camera.py --stylize_preview --capture_device 0 --vertical --fullscreen $*
done
