import datetime
import os
import json
import subprocess
import concurrent.futures

from app.frame_extraction.object_detector import ObjectDetector

N_VIDEOS_TO_PROCESS = 3
PHRASES_DIR = "LP_P1 transcrição livre"
FACIAL_EXPRESSIONS_DIR = "GLOSA_P1_EXPRESSAO"

od = ObjectDetector()


def extract_frames(videos_dir, frames_dir, annotations_dir):
    """ Extract the frames from the videos and save them in the static/videofiles/frames folder. """

    for i, video in enumerate(os.listdir(videos_dir)):
        if i >= N_VIDEOS_TO_PROCESS:  # Stop the loop after processing #N_VIDEOS_TO_PROCESS videos
            break

        video_path = os.path.join(videos_dir, video)
        videoname, extension = os.path.splitext(video)
        video_phrases_dir = os.path.join(frames_dir, PHRASES_DIR, videoname)
        video_facial_expressions_dir = os.path.join(frames_dir, FACIAL_EXPRESSIONS_DIR, videoname)
        annotation_path = os.path.join(annotations_dir, f"{videoname}.json")

        if os.path.isdir(video_phrases_dir):
            continue

        # Create the directories for the video if it doesn't exist
        os.makedirs(video_phrases_dir, exist_ok=True)
        os.makedirs(video_facial_expressions_dir, exist_ok=True)

        extract_facial_expressions_frames(video_path, video_facial_expressions_dir, annotation_path)

        extract_phrases_frames(video_path, video_phrases_dir, annotation_path)


def extract_facial_expressions_frames(video_path, facial_expressions_dir, annotation_path):
    """ Extract the facial expressions frames from the videos
    and save them in the static/videofiles/frames folder. """

    # Cycle through the annotations referring facial expressions
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

        if FACIAL_EXPRESSIONS_DIR in annotations:
            for facial_expression in annotations[FACIAL_EXPRESSIONS_DIR]["annotations"]:
                # Convert start and end time from milliseconds to hh:mm:ss format
                start_time_milliseconds = facial_expression["start_time"]
                start_time_seconds = int(start_time_milliseconds) / 1000  # convert milliseconds to seconds
                start_time_str = str(datetime.timedelta(seconds=start_time_seconds))

                end_time_milliseconds = facial_expression["end_time"]
                end_time_seconds = int(end_time_milliseconds) / 1000
                end_time_str = str(datetime.timedelta(seconds=end_time_seconds))

                annotation_id = facial_expression["annotation_id"]

                # Create a directory for the facial expression inside the video directory
                expression_dir = os.path.join(facial_expressions_dir, f"{annotation_id}")
                os.makedirs(expression_dir, exist_ok=True)

                # Extract the facial expressions frames from the video
                command = ["ffmpeg",
                           "-i", video_path,  # input file
                           "-ss", start_time_str,  # start time
                           "-to", end_time_str,  # end time
                           # "-vf", "fps=1",  # extract 1 frame per second
                           os.path.join(expression_dir, f"{annotation_id}_%02d.png")  # output file
                           ]

                subprocess.call(command)

    # Crop the extracted frames to contain only the person
    for expression_dir in os.listdir(facial_expressions_dir):
        expression_dir_path = os.path.join(facial_expressions_dir, expression_dir)

        for frame in os.listdir(expression_dir_path):
            frame_path = os.path.join(expression_dir_path, frame)
            od.detect_person(frame_path)


def extract_phrases_frames(video_path, phrases_dir, annotation_path):
    """" Extract one frame per phrase from the videos """

    # Cycle through the annotations referring facial expressions
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

        if PHRASES_DIR in annotations:
            for phrase in annotations[PHRASES_DIR]["annotations"]:
                # Convert start and end time from milliseconds to hh:mm:ss format
                start_time_milliseconds = int(phrase["start_time"])

                end_time_milliseconds = int(phrase["end_time"])

                middle_time_seconds = ((start_time_milliseconds + end_time_milliseconds) / 2) / 1000
                middle_time_str = str(datetime.timedelta(seconds=middle_time_seconds))

                annotation_id = phrase["annotation_id"]

                # Create a directory for the facial expression inside the video directory
                annotation_dir = os.path.join(phrases_dir, f"{annotation_id}")
                os.makedirs(annotation_dir, exist_ok=True)

                # Extract the phrase middle frame from the video
                command = ["ffmpeg",
                           "-i", video_path,  # input file
                           "-ss", middle_time_str,  # start time
                           "-vframes", "1",  # output one frame
                           "-update", "1",  # write a single image
                           os.path.join(annotation_dir, f"{annotation_id}.png")  # output file
                           ]

                subprocess.call(command)
