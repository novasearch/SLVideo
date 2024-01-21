#!/bin/bash

# VIDEO_PATH='../videofiles/mp4/*.mp4'
VIDEO_PATH=$1
# VIDEO_ID='*'
VIDEO_ID=$2
# RESULT_PATH='../results/'
RESULT_PATH=$3

# Extract frames from video
mkdir -p "$RESULT_PATH"/../frames/$VIDEO_ID
#ffmpeg -i "$VIDEO_PATH" -r exp9 frame%d.png # Extract frames every second
#ffmpeg -i "$VIDEO_PATH" -vf "select=eq(pict_type\,PICT_TYPE_I)" -vsync vfr "$RESULT_PATH"/../frames/$VIDEO_ID/"keyframe"%04d.jpeg
ffmpeg -skip_frame nokey -i "$VIDEO_PATH" -vsync vfr "$RESULT_PATH"/frames/"$VIDEO_ID"/"keyframe"%3d.jpeg

