import datetime
import os
import json
import subprocess
import re

VIDEOS_DIR = "app/static/videofiles/mp4"
FRAMES_DIR = "app/static/videofiles/frames"
ANNOTATIONS_DIR = "app/static/videofiles/annotations"


def extract_facial_expressions_frames():
    """ Extract the facial expressions frames from the videos
    and save them in the static/videofiles/frames folder. """

    for i, video in enumerate(os.listdir(VIDEOS_DIR)):
        if i >= 3:  # Stop the loop after processing 3 videos
            break

        video_path = os.path.join(VIDEOS_DIR, video)
        videoname, extension = os.path.splitext(video)
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{videoname}.json")
        video_dir = os.path.join(FRAMES_DIR, videoname)

        if os.path.isdir(video_dir):
            continue

        # Cycle through the annotations referring facial expressions
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
            print(annotation_path)
            annotations_keys = ["GLOSA_P1_EXPRESSÃO", "GLOSA_P1_EXPRESSAO"]

            for key in annotations_keys:
                if key in annotations:
                    for facial_expression in annotations[key]["annotations"]:
                        # Convert start and end time from milliseconds to hh:mm:ss format
                        start_time_milliseconds = facial_expression["start_time"]
                        start_time_seconds = int(start_time_milliseconds) / 1000  # convert milliseconds to seconds
                        start_time_str = str(datetime.timedelta(seconds=start_time_seconds))

                        end_time_milliseconds = facial_expression["end_time"]
                        end_time_seconds = int(end_time_milliseconds) / 1000
                        end_time_str = str(datetime.timedelta(seconds=end_time_seconds))

                        value = facial_expression["value"]

                        # Create a directory for the video if it doesn't exist
                        os.makedirs(video_dir, exist_ok=True)

                        # Create a directory for the facial expression inside the video directory
                        expression_dir = os.path.join(video_dir, f"{value}_{start_time_milliseconds}")

                        # Erase invalid characters
                        safe_expression_dir = re.sub(r'[*?:"<>|]', '', expression_dir)
                        safe_expression_value = re.sub(r'[*?:"<>|]', '', value)

                        os.makedirs(safe_expression_dir, exist_ok=True)

                        # Extract the facial expressions frames from the video
                        command = ["ffmpeg",
                                   "-i", video_path,  # input file
                                   "-ss", start_time_str,  # start time
                                   "-to", end_time_str,  # end time
                                   # "-vf", "fps=1",  # extract 1 frame per second
                                   os.path.join(safe_expression_dir, f"{safe_expression_value}_%02d.png")  # output file
                                   ]

                        subprocess.call(command)
