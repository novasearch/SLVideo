import json
import os
import datetime
from datetime import datetime as dt

from flask import Blueprint, request, render_template, flash, url_for, redirect

from .embeddings import embeddings_processing
from .frame_extraction import frames_processing
from .utils import embedder, FACIAL_EXPRESSIONS_ID, ANNOTATIONS_PATH, opensearch

bp = Blueprint('annotations', __name__)

prev_page = ""


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

    expression, start_time, end_time, phrase, converted_start_time, converted_end_time = "", "", "", "", "", ""

    for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
        if annotation["annotation_id"] == annotation_id:
            expression = annotation["value"]
            start_time = annotation["start_time"]
            end_time = annotation["end_time"]
            phrase = annotation["phrase"]

            converted_start_time = (
                    datetime.datetime.min + datetime.timedelta(seconds=int(start_time) // 1000)).time().strftime(
                "%H:%M:%S")
            converted_end_time = (
                    datetime.datetime.min + datetime.timedelta(seconds=int(end_time) // 1000)).time().strftime(
                "%H:%M:%S")

    if request.method == "POST":
        action = request.form.get("action_type")

        # If the user wants to delete the annotation
        if action == "delete":
            for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
                if annotation["annotation_id"] == annotation_id:
                    video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"].remove(annotation)

            with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
                json.dump(video_annotations, f, indent=4)

            # Delete the embeddings
            embeddings_processing.delete_embeddings(video_id, annotation_id)

            # Delete the opensearch document
            opensearch.delete_document(video_id, annotation_id)

            # Delete the frames
            frames_processing.delete_frames(video_id, annotation_id)

            return redirect(prev_page)

        # If the user wants to edit the annotation
        elif action == "edit":
            new_expression = request.form.get("expression")
            new_start_time = request.form.get("start_time")
            new_end_time = request.form.get("end_time")
            new_phrase = request.form.get("phrase")

            new_converted_start_time = convert_to_milliseconds(new_start_time)
            new_converted_end_time = convert_to_milliseconds(new_end_time)

            for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
                if annotation["annotation_id"] == annotation_id:
                    annotation["value"] = new_expression
                    annotation["start_time"] = int(new_converted_start_time)
                    annotation["end_time"] = int(new_converted_end_time)
                    annotation["phrase"] = new_phrase

            with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
                json.dump(video_annotations, f, indent=4)

            # Update the embeddings
            new_embeddings = embeddings_processing.update_annotations_embeddings(video_id, annotation_id, embedder)

            # Update the opensearch index
            opensearch.update_annotation_embedding(video_id, annotation_id, new_embeddings)

            if converted_start_time != new_start_time or converted_end_time != new_end_time:
                # Update the frames
                frames_processing.delete_frames(video_id, annotation_id)
                frames_processing.extract_annotation_frames(video_id, annotation_id, new_start_time, new_end_time)

            flash("Annotation updated successfully!", "success")

            return render_template("annotations/edit_annotation.html", video=video_id, annotation_id=annotation_id,
                                   prev_page=prev_page, expression=new_expression, start_time=new_start_time,
                                   end_time=new_end_time, phrase=new_phrase)

    return render_template("annotations/edit_annotation.html", video=video_id, annotation_id=annotation_id,
                           prev_page=prev_page, expression=expression, start_time=converted_start_time,
                           end_time=converted_end_time, phrase=phrase)


@bp.route("/add_annotation/<video_id>", methods=("GET", "POST"))
def add_annotation(video_id):
    """ Add an annotation """
    global prev_page

    current_route = url_for('annotations.add_annotation', video_id=video_id, _external=True)
    referer = request.headers.get('Referer', None)

    if (referer.replace('http://', '').replace('https://', '') !=
            current_route.replace('http://', '').replace('https://', '')):
        prev_page = referer

    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)

    last_annotation_id = video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"][-1]["annotation_id"]

    new_annotation_id = "a" + str(int(last_annotation_id.split("a")[1]) + 1)

    if request.method == "POST":
        expression = request.form.get("expression")
        start_time = request.form.get("start_time")
        end_time = request.form.get("end_time")
        phrase = request.form.get("phrase")

        converted_start_time = convert_to_milliseconds(start_time)
        converted_end_time = convert_to_milliseconds(end_time)

        annotation = {
            "annotation_id": new_annotation_id,
            "value": expression,
            "start_time": int(converted_start_time),
            "end_time": int(converted_end_time),
            "phrase": phrase,
            "user_rating": 0
        }

        if new_annotation_id not in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
            video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"].append(annotation)

            with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
                json.dump(video_annotations, f, indent=4)

            """ TODO: Uncomment this when the embeddings are ready
            # Update the embeddings
            new_embeddings = embeddings_processing.update_annotations_embeddings(video_id, new_annotation_id, embedder)

            # Index the new annotation in opensearch
            doc = opensearch.gendoc(...)
            opensearch.index_if_not_exists(doc) 

            # Extract the frames
            frames_processing.extract_annotation_frames(video_id, new_annotation_id, start_time, end_time)
            
            """

            flash("Annotation added successfully!", "success")
        else:
            flash(f"Annotation with ID {new_annotation_id} already exists!", "danger")

        return render_template("annotations/add_annotations.html", video=video_id, prev_page=prev_page,
                               new_annotation_id=new_annotation_id)

    return render_template("annotations/add_annotations.html", video=video_id, prev_page=prev_page,
                           new_annotation_id=new_annotation_id)


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
