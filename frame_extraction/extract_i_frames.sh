#!/bin/bash

# VIDEO_PATH='../videofiles/9.mp4
VIDEO_PATH=$1
# VIDEO_ID='9'
VIDEO_ID=$2
# RESULT_PATH='../results/'
RESULT_PATH=$3

# Extract frames from video
mkdir -p "$RESULT_PATH"/../frames
#ffmpeg -i "$VIDEO_PATH" -r 1 frame%d.png # Extract frames every second
ffmpeg -i "$VIDEO_PATH" -vf "select=eq(pict_type\,PICT_TYPE_I)" -vsync vfr "$RESULT_PATH"/../frames/$VIDEO_ID/"keyframe"%04d.png
