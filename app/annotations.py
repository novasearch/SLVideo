import json
import os
import datetime
import subprocess
from datetime import datetime as dt

from flask import Blueprint, request, render_template, flash, url_for, redirect

from .eaf_parser import eaf_parser
from .embeddings import embeddings_processing
from .frame_extraction import frames_processing
from .utils import embedder, FACIAL_EXPRESSIONS_ID, ANNOTATIONS_PATH, opensearch, VIDEO_PATH
from .opensearch.opensearch import gen_doc

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

    frame_rate = video_annotations["properties"]["frame_rate"]

    expression, start_time, end_time, phrase = "", "", "", ""

    for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
        if annotation["annotation_id"] == annotation_id:
            expression = annotation["value"]
            start_time = int(annotation["start_time"])
            end_time = int(annotation["end_time"])
            phrase = annotation["phrase"]

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

            # Delete the annotation from the EAF file
            eaf_parser.delete_annotation(video_id, annotation_id, FACIAL_EXPRESSIONS_ID)

            return redirect(prev_page)

        # If the user wants to edit the annotation
        elif action == "edit":
            new_expression = request.form.get("expression")
            new_phrase = request.form.get("phrase")

            start_minutes = request.form.get("start_minutes")
            start_seconds = request.form.get("start_seconds")
            start_ms = request.form.get("start_ms")
            new_start_time = convert_to_milliseconds("0", start_minutes, start_seconds, start_ms)

            end_minutes = request.form.get("end_minutes")
            end_seconds = request.form.get("end_seconds")
            end_ms = request.form.get("end_ms")
            new_end_time = convert_to_milliseconds("0", end_minutes, end_seconds, end_ms)

            parent_ref = None

            for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
                if annotation["annotation_id"] == annotation_id:
                    annotation["value"] = new_expression
                    annotation["start_time"] = str(new_start_time)
                    annotation["end_time"] = str(new_end_time)
                    annotation["phrase"] = new_phrase
                    parent_ref = annotation.get("annotation_ref", None)

            if start_time != new_start_time or end_time != new_end_time:
                # Update parent annotation if there is a parent tier
                if parent_ref is not None:
                    parent_tier = video_annotations[FACIAL_EXPRESSIONS_ID].get("parent_ref", None)
                    for annotation in video_annotations[parent_tier]["annotations"]:
                        if annotation["annotation_id"] == parent_ref:
                            annotation["start_time"] = str(new_start_time)
                            annotation["end_time"] = str(new_end_time)

                # Update the frames
                frames_processing.delete_frames(video_id, annotation_id)
                update_embeddings_and_index(video_id, annotation_id, start_time, end_time)

            else:
                # Update the annotation's embeddings
                new_embeddings = embeddings_processing.update_annotations_embeddings(video_id, annotation_id, embedder)

                # Update the opensearch index
                opensearch.update_annotation_embedding(video_id, annotation_id, new_embeddings)

            # Update the EAF file
            eaf_parser.edit_annotation(video_id, FACIAL_EXPRESSIONS_ID, annotation_id, new_start_time,
                                       new_end_time, new_expression)

            with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
                json.dump(video_annotations, f, indent=4)

            flash("Annotation updated successfully!", "success")

            return render_template("annotations/edit_annotation.html", video=video_id, annotation_id=annotation_id,
                                   prev_page=prev_page, expression=new_expression, start_time=new_start_time,
                                   end_time=new_end_time, phrase=new_phrase, frame_rate=frame_rate)

    return render_template("annotations/edit_annotation.html", video=video_id, annotation_id=annotation_id,
                           prev_page=prev_page, expression=expression, start_time=start_time,
                           end_time=end_time, phrase=phrase, frame_rate=frame_rate)


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

    frame_rate = video_annotations["properties"]["frame_rate"]
    new_annotation_id = "a" + str(int(video_annotations["properties"]["lastUsedAnnotationId"]) + 1)

    if request.method == "POST":
        start_minutes = request.form.get("start_minutes")
        start_seconds = request.form.get("start_seconds")
        start_ms = request.form.get("start_ms")
        new_start_time = convert_to_milliseconds("0", start_minutes, start_seconds, start_ms)

        end_minutes = request.form.get("end_minutes")
        end_seconds = request.form.get("end_seconds")
        end_ms = request.form.get("end_ms")
        new_end_time = convert_to_milliseconds("0", end_minutes, end_seconds, end_ms)

        expression = request.form.get("expression")
        phrase = request.form.get("phrase")

        if new_annotation_id not in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
            # Add respective parent annotation if there is a parent tier
            parent_tier = video_annotations[FACIAL_EXPRESSIONS_ID].get("parent_ref", None)
            if parent_tier:
                parent_id = new_annotation_id
                new_annotation_id = "a" + str(int(new_annotation_id.split("a")[1]) + 1)
                parent_annotation = {
                    "annotation_id": parent_id,
                    "value": expression,
                    "start_time": int(new_start_time),
                    "end_time": int(new_end_time)
                }
                video_annotations[parent_tier]["annotations"].append(parent_annotation)

                annotation = {
                    "annotation_id": new_annotation_id,
                    "annotation_ref": parent_id,  # Reference to the parent annotation
                    "value": expression,
                    "start_time": int(new_start_time),
                    "end_time": int(new_end_time),
                    "phrase": phrase,
                    "user_rating": 0
                }
            else:
                annotation = {
                    "annotation_id": new_annotation_id,
                    "value": expression,
                    "start_time": int(new_start_time),
                    "end_time": int(new_end_time),
                    "phrase": phrase,
                    "user_rating": 0
                }

            video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"].append(annotation)
            video_annotations["properties"]["lastUsedAnnotationId"] = new_annotation_id.split("a")[1]

            update_embeddings_and_index(video_id, new_annotation_id, new_start_time, new_end_time)

            # Update the EAF file
            eaf_parser.add_annotation(video_id, new_annotation_id, FACIAL_EXPRESSIONS_ID, new_start_time,
                                      new_end_time, expression, phrase)

            with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "w") as f:
                json.dump(video_annotations, f, indent=4)

            flash("Annotation added successfully!", "success")
        else:
            flash(f"Annotation with ID {new_annotation_id} already exists!", "danger")

        return render_template("annotations/add_annotations.html", video=video_id, prev_page=prev_page,
                               annotation_id=new_annotation_id, frame_rate=frame_rate)

    return render_template("annotations/add_annotations.html", video=video_id, prev_page=prev_page,
                           annotation_id=new_annotation_id, frame_rate=frame_rate)


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


def update_embeddings_and_index(video_id, new_annotation_id, start_time, end_time):
    """ Update the frames, embeddings and index the new annotation """
    # Convert the start and end time from milliseconds to HH:MM:SS.MS format
    start_time = str(dt.utcfromtimestamp(start_time / 1000).strftime('%H:%M:%S.%f')[:-3])
    end_time = str(dt.utcfromtimestamp(end_time / 1000).strftime('%H:%M:%S.%f')[:-3])

    # Extract the frames
    frames_processing.extract_annotation_frames(video_id, new_annotation_id, start_time, end_time)

    # Generate the embeddings
    embeddings_processing.add_embeddings(video_id, new_annotation_id, embedder)

    # Load the embeddings
    (base_frame_embeddings, average_frame_embeddings, best_frame_embeddings, summed_frame_embeddings,
     annotations_embeddings) = embeddings_processing.load_embeddings()

    # Index the new annotation in opensearch
    doc = gen_doc(
        video_id=video_id,
        annotation_id=new_annotation_id,
        base_frame_embedding=base_frame_embeddings[video_id][new_annotation_id].tolist(),
        average_frame_embedding=average_frame_embeddings[video_id][new_annotation_id].tolist(),
        best_frame_embedding=best_frame_embeddings[video_id][new_annotation_id].tolist(),
        summed_frame_embeddings=summed_frame_embeddings[video_id][new_annotation_id].tolist(),
        annotation_embedding=annotations_embeddings[video_id][new_annotation_id].tolist(),
    )
    opensearch.index_if_not_exists(doc)


def convert_to_milliseconds(hours, minutes, seconds, milliseconds):
    """ Convert the time to milliseconds """
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)



