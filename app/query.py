import datetime
import json
import os
import subprocess
import time

import numpy as np
from .embeddings import generate_embeddings

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from .opensearch.opensearch import LGPOpenSearch

FRAMES_PATH = "app/static/videofiles/frames"
ANNOTATIONS_PATH = "app/static/videofiles/annotations"
VIDEO_CLIP_PATH = "app/static/videofiles/mp4/videoclip.mp4"

PHRASES_ID = "LP_P1 transcrição livre"
FACIAL_EXPRESSIONS_ID = "GLOSA_P1_EXPRESSAO"

N_RESULTS = 30

bp = Blueprint('query', __name__)
opensearch = LGPOpenSearch()


@bp.route("/", methods=("GET", "POST"))
def query():
    """Query for a video"""
    if request.method == "POST":
        query_input = request.form["query"]
        selected_field = int(request.form.get('field'))
        session['search_mode'] = request.form.get('mode')
        session['similarity_scores'] = {}
        error = None

        if not query_input:
            error = "Query is required."

        if error is not None:
            flash(error)
        else:

            if selected_field == 1:  # Base Frames Embeddings
                session['query_results'] = query_frames_embeddings(query_input)
            elif selected_field == 2:  # Average Frames Embeddings
                session['query_results'] = query_average_frames_embeddings(query_input)
            elif selected_field == 3:  # Best Frame Embedding
                session['query_results'] = query_best_frame_embedding(query_input)
            elif selected_field == 4:  # True Expression
                session['query_results'] = query_true_expression(query_input)

            return redirect(url_for("query.videos_results"))

    return render_template("query/query.html")


@bp.route("/videos_results", methods=("GET", "POST"))
def videos_results():
    """Display results of query."""
    query_results = session.get('query_results', {})
    search_mode = session.get('search_mode', 1)

    frames = {}
    videos_info = {}

    # Get a frame for each video
    for video_id in query_results:
        annotations = query_results[video_id]

        videos_info[video_id] = {}
        videos_info[video_id]['video_name'] = video_id
        videos_info[video_id]['n_annotations'] = len(annotations)
        videos_info[video_id]['first_annotation'] = annotations[0]

        frames_path = os.path.join(FRAMES_PATH, search_mode, video_id, annotations[0])
        frames[video_id] = os.listdir(frames_path)[0]

    if request.method == "POST":
        selected_video = request.form.get("selected_video")
        error = None

        if not selected_video:
            error = "Selecting a video is required."

        if error is not None:
            flash(error)
        else:
            return redirect(url_for("query.clips_results", video=selected_video))

    return render_template("query/videos_results.html", frames=frames, videos_info=videos_info, search_mode=search_mode)


@bp.route("/clips_results/<video>", methods=("GET", "POST"))
def clips_results(video):
    """Display results of query."""
    query_results = session['query_results'][video]
    search_mode = session.get('search_mode', 1)

    frames = {}
    frames_info = {}

    # Get the searched frames
    for annotation_id in query_results:
        annotation_path = os.path.join(ANNOTATIONS_PATH, f"{video}.json")
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
            for annotation in annotations[search_mode]["annotations"]:
                if annotation["annotation_id"] == annotation_id:
                    frames_path = os.path.join(FRAMES_PATH, search_mode, video, annotation_id)
                    frames[annotation_id] = os.listdir(frames_path)
                    frames_info[annotation_id] = annotation
                    converted_start_time = str(datetime.timedelta(seconds=int(annotation["start_time"]) // 1000))
                    frames_info[annotation_id]["converted_start_time"] = converted_start_time
                    frames_info[annotation_id]["similarity_score"] = session['similarity_scores'][
                        video + "_" + annotation_id]
                    break

    if search_mode == FACIAL_EXPRESSIONS_ID:
        # Not all frames of each expression are going to be displayed
        frames_to_display = {}
        num_frames_to_display = 5

        # Display only #num_frames_to_display frames of each expression
        for expression, all_frames in frames.items():

            # Calculate the step size
            if len(all_frames) <= num_frames_to_display:
                step_size = 1
            else:
                step_size = (len(all_frames) - 1) // num_frames_to_display + 1

            # Select frames to display
            frames_to_display[expression] = all_frames[::step_size]

            # Ensure that only num_frames_to_display frames are selected
            frames_to_display[expression] = frames_to_display[expression][:num_frames_to_display]
    else:
        # If the frames are not the facial expression's ones then it just has one frame per result
        frames_to_display = frames

    if request.method == "POST":
        selected_annotation = request.form.get("selected_annotation")
        annotation = frames_info[selected_annotation]
        error = None

        if not selected_annotation:
            error = "Expression is required."

        if error is not None:
            flash(error)
        else:
            session['annotation'] = annotation
            return redirect(url_for("query.play_selected_result", video=video, annotation_id=selected_annotation))

    return render_template("query/clips_results.html", frames=frames_to_display, frames_info=frames_info,
                           search_mode=search_mode, video=video)


@bp.route("/results/<video>/<annotation_id>", methods=("GET", "POST"))
def play_selected_result(video, annotation_id):
    """ Display the video clip of the selected expression. """

    annotation = session.get('annotation')

    # Convert the timestamp to seconds
    init_time = int(annotation["start_time"]) // 1000
    end_time = int(annotation["end_time"]) // 1000

    return render_template("query/play_expression.html", value=annotation["value"], video=video,
                           init_time=init_time, end_time=end_time)


def query_frames_embeddings(query_input):
    query_embedding = generate_embeddings.generate_query_embeddings(query_input)

    search_results = opensearch.knn_query(query_embedding.tolist(), N_RESULTS)
    query_results = {}

    for hit in search_results['hits']['hits']:
        if hit['_source']['video_id'] not in query_results:
            query_results[hit['_source']['video_id']] = []
        query_results[hit['_source']['video_id']].append(hit['_source']['annotation_id'])
        session['similarity_scores'][hit['_id']] = hit['_score']

    print_performance_metrics(query_results, query_input)

    session['query_results'] = query_results

    return query_results


def query_average_frames_embeddings(query_input):
    query_embedding = generate_embeddings.generate_query_embeddings(query_input)

    search_results = opensearch.knn_query_average(query_embedding.tolist(), N_RESULTS)
    query_results = {}

    for hit in search_results['hits']['hits']:
        if hit['_source']['video_id'] not in query_results:
            query_results[hit['_source']['video_id']] = []
        query_results[hit['_source']['video_id']].append(hit['_source']['annotation_id'])
        session['similarity_scores'][hit['_id']] = hit['_score']

    print_performance_metrics(query_results, query_input)

    session['query_results'] = query_results

    return query_results


def query_best_frame_embedding(query_input):
    query_embedding = generate_embeddings.generate_query_embeddings(query_input)

    search_results = opensearch.knn_query_best(query_embedding.tolist(), N_RESULTS)
    query_results = {}

    for hit in search_results['hits']['hits']:
        if hit['_source']['video_id'] not in query_results:
            query_results[hit['_source']['video_id']] = []
        query_results[hit['_source']['video_id']].append(hit['_source']['annotation_id'])
        session['similarity_scores'][hit['_id']] = hit['_score']

    print_performance_metrics(query_results, query_input)

    session['query_results'] = query_results

    return query_results


def query_true_expression(query_input):
    """ Get the results of the query using the ground truth """
    query_results = {}
    search_mode = session.get('search_mode', 1)

    for video in os.listdir(os.path.join(FRAMES_PATH, search_mode)):
        with open(os.path.join(ANNOTATIONS_PATH, f"{video}.json"), "r") as f:
            video_annotations = json.load(f)
            if search_mode in video_annotations:
                for annotation in video_annotations[search_mode]["annotations"]:
                    if annotation["value"] is not None and query_input.lower() in annotation["value"].lower():
                        if video not in query_results:
                            query_results[video] = []
                        query_results[video].append(annotation["annotation_id"])
                        session['similarity_scores'][annotation["annotation_id"]] = "N/A"

    session['query_results'] = query_results
    return query_results


def print_performance_metrics(query_results, query_input):
    compare_results = query_true_expression(query_input)

    # Initialize counters for true positives, false positives and false negatives
    tp = 0
    fp = 0
    fn = 0

    # Iterate over each video in query_results
    for video, query_annotations in query_results.items():
        # If the video is also in compare_results
        if video in compare_results:
            compare_annotations = compare_results[video]

            # Count the number of true positives, false positives and false negatives
            tp += len(set(query_annotations).intersection(compare_annotations))
            fp += len(set(query_annotations).difference(compare_annotations))
            fn += len(set(compare_annotations).difference(query_annotations))
        else:
            # If the video is not in compare_results, all annotations are false positives
            fp += len(query_annotations)

    # Add the false negatives for videos only in compare_results
    for video, compare_annotations in compare_results.items():
        if video not in query_results:
            fn += len(compare_annotations)

    # Calculate precision, recall and F1 score
    precision = round(tp / (tp + fp), 2) if tp + fp > 0 else 0.0
    recall = round(tp / (tp + fn), 2) if tp + fn > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 2) if precision + recall > 0 else 0.0

    print("-------------------------------------")
    print("-------------------------------------")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("-------------------------------------")
    print("-------------------------------------")
