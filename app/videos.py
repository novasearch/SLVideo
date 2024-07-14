import json
import os
import datetime

from flask import Blueprint, render_template, request, redirect, url_for

from app.utils import VIDEO_PATH, FRAMES_PATH, PHRASES_ID, ANNOTATIONS_PATH, FACIAL_EXPRESSIONS_ID

bp = Blueprint('videos', __name__)


@bp.route('/videos', methods=("GET", "POST"))
def list_videos():
    """ Shows all the videos in the database. """

    videos = {}
    for video in os.listdir(VIDEO_PATH):
        if not video.endswith(".mp4"):
            continue
        video_name = video.split('.')[0]
        frames_path = os.path.join(FRAMES_PATH, PHRASES_ID, video_name)
        first_annotation_path = os.listdir(frames_path)[0]
        thumbnail = os.listdir(os.path.join(frames_path, first_annotation_path))[0]

        videos[video_name] = {}
        videos[video_name]['path'] = os.path.join(VIDEO_PATH, video)
        videos[video_name]['thumbnail'] = os.path.join(PHRASES_ID, video_name, first_annotation_path, thumbnail)

    if request.method == "POST":
        video_id = request.form.get("selected_video")
        return redirect(url_for("videos.watch_video", video_id=video_id))

    return render_template("videos_list/videos_list.html", videos=videos)


@bp.route('/videos/<video_id>', methods=("GET", "POST"))
def watch_video(video_id):
    """ Shows the selected video. """
    facial_expressions = {}

    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        videos_annotations = json.load(f)

        if FACIAL_EXPRESSIONS_ID in videos_annotations:
            for annotation in videos_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
                facial_expressions[annotation["annotation_id"]] = {}
                facial_expressions[annotation["annotation_id"]]["value"] = annotation["value"]
                facial_expressions[annotation["annotation_id"]]["start_time"] = round(
                    int(annotation["start_time"]) // 1000)
                facial_expressions[annotation["annotation_id"]]["end_time"] = round(
                    int(annotation["end_time"]) // 1000)

    # Sort annotations by value alphabetically
    facial_expressions = dict(sorted(facial_expressions.items(), key=lambda item: item[1]["value"]))

    if request.method == "POST":
        form_type = request.form.get("form_type")
        if form_type == "add":
            return redirect(url_for("annotations.add_annotation", video_id=video_id))
        elif form_type == "edit":
            annotation_id = request.form.get("selected_annotation")
            return redirect(url_for("annotations.edit_annotation", video_id=video_id, annotation_id=annotation_id))

    return render_template("videos_list/watch_video.html", video=video_id,
                           annotations=facial_expressions)
