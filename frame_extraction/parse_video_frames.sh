#!/bin/bash

# VIDEO_DIR='/user/data/j.bordalo/howto100m/videos_small'
VIDEO_DIR=$1
# RESULT_PATH=../frames/
RESULT_PATH=$2

VIDEO_ID="${video%.*}"
if [[ $(find $RESULT_PATH -type f -name "*$VIDEO_ID*") ]]; then
  echo "$VIDEO_ID in out directory"
  exit 1
fi

echo "Processing video $VIDEO_ID"

VIDEO_PATH=$VIDEO_DIR"/"$video
./extract_i_frames.sh $VIDEO_PATH $VIDEO_ID $RESULT_PATH

# Get timestamps
mkdir -p $RESULT_PATH/../timestamps
OUTPUT_FILE=$RESULT_PATH/../timestamps/$VIDEO_ID.json
echo "{ " > $OUTPUT_FILE
ffprobe -skip_frame nokey -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 $VIDEO_PATH | awk -v count=1 -F',' '/K/ {printf("\"%s-f%03d\": %s,\n", video_id, count, $1) ; count++}' video_id=$VIDEO_ID  >> $OUTPUT_FILE
sed -i '$ s/.$//' $OUTPUT_FILE # remove last ,
echo "}" >> $OUTPUT_FILE

