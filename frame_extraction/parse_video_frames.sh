#!/bin/bash

# VIDEO_DIR='../videofiles/'
VIDEO_DIR=$1
# RESULT_PATH='../results/'
RESULT_PATH=$2

VIDEO_ID=$(basename "${VIDEO_DIR}" .mp4)
VIDEO_ID=$(basename "${VIDEO_ID}")

if [[ $(find $RESULT_PATH -type f -name "*$VIDEO_ID*") ]]; then
  echo "$VIDEO_ID in out directory"
  exit 1
fi

echo "Processing video $VIDEO_ID"

VIDEO_PATH=$VIDEO_DIR"/"$VIDEO_ID
./extract_i_frames.sh $VIDEO_PATH $VIDEO_ID $RESULT_PATH

FRAMES_PATH="$RESULT_PATH"/../frames/$VIDEO_ID/

# Get timestamps
mkdir -p "$RESULT_PATH"/../timestamps
OUTPUT_FILE=$RESULT_PATH/../timestamps/$VIDEO_ID"_timestamps".json
echo "{ " > "$OUTPUT_FILE"
ffprobe -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 "$FRAMES_PATH" | awk -v count=1 -F',' '/K/ {printf("\"%s-f%03d\": %s,\n", video_id, count, $1) ; count++}' video_id=$VIDEO_ID  >> $OUTPUT_FILEsed -i '$ s/.$//' "$OUTPUT_FILE" # remove last ,
echo "}" >> "$OUTPUT_FILE"

