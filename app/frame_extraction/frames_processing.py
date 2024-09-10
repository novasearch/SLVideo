import datetime
import os
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from app.utils import PHRASES_ID, FACIAL_EXPRESSIONS_ID, VIDEO_PATH, FRAMES_PATH, ANNOTATIONS_PATH
from app.frame_extraction import object_detector

THREAD_COUNT = 4


class FrameExtractor:

    def __init__(self):
        self.od = object_detector.ObjectDetector()

    def extract_frames(self):
        """ Extract the frames from the videos and save them in the static/videofiles/frames folder. """

        for i, video in enumerate(os.listdir(VIDEO_PATH)):
            # Stop the loop after processing #N_VIDEOS_TO_PROCESS videos
            # if i >= N_VIDEOS_TO_PROCESS:
            #     break

            # Define the paths for the video, phrases frames and facial expressions frames
            video_path = os.path.join(VIDEO_PATH, video)
            videoname, extension = os.path.splitext(video)
            video_phrases_dir = os.path.join(FRAMES_PATH, PHRASES_ID, videoname)
            video_facial_expressions_dir = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID, videoname)
            annotation_path = os.path.join(ANNOTATIONS_PATH, f"{videoname}.json")

            if not os.path.isdir(video_facial_expressions_dir):
                os.makedirs(video_facial_expressions_dir, exist_ok=True)
                self.extract_facial_expressions_frames(video_path, video_facial_expressions_dir, annotation_path)

            if not os.path.isdir(video_phrases_dir):
                os.makedirs(video_phrases_dir, exist_ok=True)
                self.extract_phrases_frames(video_path, video_phrases_dir, annotation_path)

    def extract_facial_expressions_frames(self, video_path, facial_expressions_dir, annotation_path):
        """ Extract the facial expressions frames from the videos
        and save them in the static/videofiles/frames folder, using
        threading to speed up the process.
        """

        # Cycle through the annotations referring facial expressions
        with open(annotation_path, "r") as f:
            annotations = json.load(f)

            if FACIAL_EXPRESSIONS_ID in annotations:
                with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
                    futures = []
                    for facial_expression in annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
                        future = executor.submit(self.process_facial_expression, video_path, facial_expressions_dir,
                                                 facial_expression)
                        futures.append(future)

                    # Wait for all futures to complete
                    for future in futures:
                        future.result()

    def process_facial_expression(self, video_path, facial_expressions_dir, facial_expression):
        """ Extract the frames of a facial expression from the video. """

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

        video_name, _ = os.path.splitext(os.path.basename(video_path))

        # Extract the facial expressions frames from the video
        command = ["ffmpeg",
                   "-loglevel", "error",  # suppress the output
                   "-ss", start_time_str,  # start time
                   "-to", end_time_str,  # end time
                   "-i", video_path,  # input file
                   # "-vf", "fps=1",  # extract 1 frame per second
                   os.path.join(expression_dir, f"{annotation_id}_%02d.png")  # output file
                   ]

        print(video_name, "-", facial_expression["annotation_id"], "|| Extraction Started", flush=True)
        subprocess.call(command)
        print(video_name, "-", facial_expression["annotation_id"], "|| Extraction Finished", flush=True)

        # Prepend the directory path to each filename in expression_dir
        images_paths = [os.path.join(expression_dir, image_path) for image_path in os.listdir(expression_dir)]

        # Crop the extracted frames to contain only the person
        self.od.detect_person(images_paths)

    def extract_phrases_frames(self, video_path, phrases_dir, annotation_path):
        """" Extract one frame per phrase from the videos """

        # Cycle through the annotations referring facial expressions
        with open(annotation_path, "r") as f:
            annotations = json.load(f)

            if PHRASES_ID in annotations:
                for phrase in annotations[PHRASES_ID]["annotations"]:
                    # Convert start and end time from milliseconds to hh:mm:ss format
                    start_time_milliseconds = int(phrase["start_time"])

                    end_time_milliseconds = int(phrase["end_time"])

                    middle_time_seconds = ((start_time_milliseconds + end_time_milliseconds) / 2) / 1000
                    middle_time_str = str(datetime.timedelta(seconds=middle_time_seconds))

                    annotation_id = phrase["annotation_id"]

                    # Create a directory for the facial expression inside the video directory
                    annotation_dir = os.path.join(phrases_dir, f"{annotation_id}")
                    os.makedirs(annotation_dir, exist_ok=True)

                    print("Extracting phrases frames from video", video_path, "of annotation",
                          phrase["annotation_id"],
                          flush=True)

                    # Extract the phrase middle frame from the video
                    command = ["ffmpeg",
                               "-loglevel", "error",  # suppress the output
                               "-ss", middle_time_str,  # start time
                               "-i", video_path,  # input file
                               "-vframes", "1",  # output one frame
                               "-update", "1",  # write a single image
                               os.path.join(annotation_dir, f"{annotation_id}.png")  # output file
                               ]

                    subprocess.call(command)


def extract_annotation_frames(video_id, annotation_id, start_time, end_time):
    """ Extract the frames of a facial expression from the video in the specified time range."""

    # Create a directory for the facial expression inside the video directory
    video_facial_expressions_dir = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID, video_id)
    expression_dir = os.path.join(video_facial_expressions_dir, f"{annotation_id}")
    os.makedirs(expression_dir, exist_ok=True)

    # Extract the facial expressions frames from the video
    command = ["ffmpeg",
               "-loglevel", "error",  # suppress the output
               "-ss", start_time,  # start time in HH:MM:SS format
               "-to", end_time,  # end time in HH:MM:SS format
               "-i", os.path.join(VIDEO_PATH, video_id + ".mp4"),  # input file
               # "-vf", "fps=1",  # extract 1 frame per second
               os.path.join(expression_dir, f"{annotation_id}_%02d.png")  # output file
               ]

    print(video_id, "-", annotation_id, "|| Extraction Started", flush=True)
    subprocess.call(command)
    print(video_id, "-", annotation_id, "|| Extraction Finished", flush=True)

    # Prepend the directory path to each filename in expression_dir
    images_paths = [os.path.join(expression_dir, image_path) for image_path in os.listdir(expression_dir)]

    # Crop the extracted frames to contain only the person
    od = object_detector.ObjectDetector()
    od.detect_person(images_paths)


def delete_frames(video_id, annotation_id):
    """ Delete the frames of a facial expression from the annotation in the video. """

    video_facial_expressions_dir = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID, video_id)
    expression_dir = os.path.join(video_facial_expressions_dir, f"{annotation_id}")

    if os.path.exists(expression_dir):
        shutil.rmtree(expression_dir)
