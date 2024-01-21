import datetime
import os
import json
import subprocess

VIDEOS_DIR = "app/static/videofiles/mp4"
FRAMES_DIR = "app/static/videofiles/frames"
ANNOTATIONS_DIR = "app/static/videofiles/annotations"


def extract_facial_expressions_frames():
    """ Extract the facial expressions frames from the videos
    and save them in the static/videofiles folder. """

    for video in os.listdir(VIDEOS_DIR):
        video_path = os.path.join(VIDEOS_DIR, video)
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{video}.json")

        # Initialize a counter for each facial expression
        expression_counters = {}

        # Cycle through the annotations referring facial expressions
        with open(annotation_path, "r") as f:
            annotations = json.load(f)

            for facial_expression in annotations["GLOSA_P1_EXPRESSAO"]["annotations"]:
                # Convert start and end time from milliseconds to hh:mm:ss format
                start_time_milliseconds = facial_expression["start_time"]
                start_time_seconds = start_time_milliseconds / 1000  # convert milliseconds to seconds
                start_time_str = str(datetime.timedelta(seconds=start_time_seconds))

                end_time_milliseconds = facial_expression["end_time"]
                end_time_seconds = end_time_milliseconds / 1000
                end_time_str = str(datetime.timedelta(seconds=end_time_seconds))

                value = facial_expression["value"]

                # Increment the counter for this facial expression
                if value not in expression_counters:
                    expression_counters[value] = 1
                else:
                    expression_counters[value] += 1

                # Create a directory for the facial expression if it doesn't exist
                expression_dir = os.path.join(FRAMES_DIR, value)
                os.makedirs(expression_dir, exist_ok=True)

                # Create a directory for the video inside the facial expression directory
                video_dir = os.path.join(expression_dir, f"{video}_{expression_counters[value]}")
                os.makedirs(video_dir, exist_ok=True)

                # Extract the facial expressions frames from the video
                command = ["ffmpeg",
                           "-i", video_path,  # input file
                           "-ss", start_time_str,  # start time
                           "-to", end_time_str,  # end time
                           "-vf", "fps=1",  # extract 1 frame per second
                           os.path.join(FRAMES_DIR, f"{video}_{value}_%d.png")  # output file
                           ]

                subprocess.call(command)
