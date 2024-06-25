import json
import os

from app.constants import ANNOTATIONS_PATH, FACIAL_EXPRESSIONS_ID


def updated_user_rating(video_id, annotation_id, rating):
    """ Update the user rating of an annotation """
    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)

    if FACIAL_EXPRESSIONS_ID not in video_annotations:
        return

    annotations = video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]

    for annotation in annotations:
        if annotation["annotation_id"] == annotation_id:
            annotation["user_rating"] = rating

    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
        json.dump(video_annotations, f, indent=4)
