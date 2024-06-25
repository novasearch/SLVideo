import json
import os

from flask import Blueprint, request, render_template

from app.constants import ANNOTATIONS_PATH, FACIAL_EXPRESSIONS_ID

bp = Blueprint('annotations', __name__)


@bp.route("/edit_annotation/<video_id>/<annotation_id>", methods=("GET", "POST"))
def edit_annotation(video_id, annotation_id):
    """ Edit an annotation """
    
    prev_page = request.headers.get('Referer', None)

    return render_template("annotations/edit_annotation.html", video=video_id, annotation_id=annotation_id, prev_page=prev_page)


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
