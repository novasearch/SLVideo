import json
import os
import datetime
from datetime import datetime as dt

from flask import Blueprint, request, render_template, flash, url_for

from app.constants import ANNOTATIONS_PATH, FACIAL_EXPRESSIONS_ID

bp = Blueprint('annotations', __name__)

prev_page = None


@bp.route("/edit_annotation/<video_id>/<annotation_id>", methods=("GET", "POST"))
def edit_annotation(video_id, annotation_id):
    """ Edit an annotation """
    global prev_page

    current_route = url_for('annotations.edit_annotation', video_id=video_id, annotation_id=annotation_id,
                            _external=True)
    referer = request.headers.get('Referer', None)

    if (referer.replace('http://', '').replace('https://', '') !=
            current_route.replace('http://', '').replace('https://', '')):
        prev_page = referer

    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)

    expression, start_time, end_time, phrase = "", "", "", ""

    for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
        if annotation["annotation_id"] == annotation_id:
            expression = annotation["value"]
            start_time = annotation["start_time"]
            end_time = annotation["end_time"]
            phrase = annotation["phrase"]

    converted_start_time = (
            datetime.datetime.min + datetime.timedelta(seconds=int(start_time) // 1000)).time().strftime("%H:%M:%S")
    converted_end_time = (datetime.datetime.min + datetime.timedelta(seconds=int(end_time) // 1000)).time().strftime(
        "%H:%M:%S")

    if request.method == "POST":
        expression = request.form.get("expression")
        start_time = convert_to_milliseconds(request.form.get("start_time"))
        end_time = convert_to_milliseconds(request.form.get("end_time"))
        phrase = request.form.get("phrase")

        for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
            if annotation["annotation_id"] == annotation_id:
                annotation["value"] = expression
                annotation["start_time"] = int(start_time)
                annotation["end_time"] = int(end_time)
                annotation["phrase"] = phrase

        with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
            json.dump(video_annotations, f, indent=4)

        flash("Annotation updated successfully!", "success")

        return render_template("annotations/edit_annotation.html", video=video_id, annotation_id=annotation_id,
                               prev_page=prev_page, expression=expression, start_time=converted_start_time,
                               end_time=converted_end_time, phrase=phrase)

    return render_template("annotations/edit_annotation.html", video=video_id, annotation_id=annotation_id,
                           prev_page=prev_page, expression=expression, start_time=converted_start_time,
                           end_time=converted_end_time, phrase=phrase)


@bp.route("/update_user_rating", methods=["POST"])
def updated_user_rating():
    """ Update the user rating of an annotation """
    data = request.get_json()
    video_id = data["video_id"]
    annotation_id = data["annotation_id"]
    rating = data["rating"]

    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)

    if FACIAL_EXPRESSIONS_ID not in video_annotations:
        return

    for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
        if annotation["annotation_id"] == annotation_id:
            annotation["user_rating"] = rating

    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
        json.dump(video_annotations, f, indent=4)

    return '', 204


def convert_to_milliseconds(time_str):
    # Convert the time string to a datetime object
    time_obj = dt.strptime(time_str, '%H:%M:%S')

    # Calculate the total milliseconds
    milliseconds = (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000

    return milliseconds
