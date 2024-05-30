import datetime
import json
import os
import re

from .embeddings import embeddings_processing

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from .opensearch.opensearch import LGPOpenSearch

# Define the paths for the frames and annotations
FRAMES_PATH = "app/static/videofiles/frames"
ANNOTATIONS_PATH = "app/static/videofiles/annotations"
VIDEO_CLIP_PATH = "app/static/videofiles/mp4/videoclip.mp4"

PHRASES_ID = "LP_P1 transcrição livre" # Annotation field for the phrases
FACIAL_EXPRESSIONS_ID = "GLOSA_P1_EXPRESSAO" # Annotation field for the facial expressions

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
        error = None

        if not query_input:
            error = "Query is required."

        if error is not None:
            flash(error)
        else:

            if selected_field == 1:  # Base Frames Embeddings
                query_frames_embeddings(query_input)
            elif selected_field == 2:  # Average Frames Embeddings
                query_average_frames_embeddings(query_input)
            elif selected_field == 3:  # Best Frame Embedding
                query_best_frame_embedding(query_input)
            elif selected_field == 4:  # True Expression / Ground Truth
                query_true_expression(query_input)

            return redirect(url_for("query.videos_results"))

    return render_template("query/query.html")


@bp.route("/videos_results", methods=("GET", "POST"))
def videos_results():
    """ Display the videos that contain the query results """
    query_results = session.get('query_results', {})
    search_mode = session.get('search_mode', 1)

    frames = {}
    videos_info = {}

    # Collect information about the retrieved videos
    for video_id in query_results:
        annotations = query_results[video_id]
        first_annotation = list(annotations.keys())[0]

        videos_info[video_id] = {}
        videos_info[video_id]['video_name'] = video_id
        videos_info[video_id]['n_annotations'] = len(annotations)
        videos_info[video_id]['first_annotation'] = first_annotation

        # Get the first frame of the first annotation to display in the results page
        frames_path = os.path.join(FRAMES_PATH, search_mode, video_id, first_annotation)
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
    """ Display the video segments of one video that contain the query results """
    query_results = session['query_results'][video]
    search_mode = session.get('search_mode', 1)

    frames = {}
    frames_info = {}

    # Collect information about the retrieved video segments
    for annotation_id in query_results.keys():
        frames_path = os.path.join(FRAMES_PATH, search_mode, video, annotation_id)
        frames[annotation_id] = os.listdir(frames_path)
        frames_info[annotation_id] = query_results[annotation_id]
        converted_start_time = str(datetime.timedelta(seconds=int(query_results[annotation_id]["start_time"]) // 1000))
        frames_info[annotation_id]["converted_start_time"] = converted_start_time
        converted_end_time = str(datetime.timedelta(seconds=int(query_results[annotation_id]["end_time"]) // 1000))
        frames_info[annotation_id]["converted_end_time"] = converted_end_time
        frames_info[annotation_id]["similarity_score"] = query_results[annotation_id]["similarity_score"]

    if search_mode == FACIAL_EXPRESSIONS_ID:
        # Not all frames of each expression are going to be displayed
        frames_to_display = {}
        num_frames_to_display = 6

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

    return render_template("query/clips_results_modal.html", frames=frames_to_display, frames_info=frames_info,
                           search_mode=search_mode, video=video)


@bp.route("/results/<video>/<annotation_id>", methods=("GET", "POST"))
def play_selected_result(video, annotation_id):
    """
    @deprecated
    Display the video clip of the selected expression
    """

    annotation = session.get('annotation')

    # Convert the timestamp to seconds
    init_time = int(annotation["start_time"]) // 1000
    end_time = int(annotation["end_time"]) // 1000

    return render_template("query/play_expression.html", value=annotation["value"], video=video,
                           init_time=init_time, end_time=end_time)


def query_frames_embeddings(query_input):
    """ Get the results of the query using the frames embeddings """
    query_embedding = generate_embeddings.generate_query_embeddings(query_input)
    search_results = opensearch.knn_query(query_embedding.tolist(), N_RESULTS)
    set_query_results(search_results, query_input)


def query_average_frames_embeddings(query_input):
    """ Get the results of the query using the average of the frames embeddings """
    query_embedding = generate_embeddings.generate_query_embeddings(query_input)
    search_results = opensearch.knn_query_average(query_embedding.tolist(), N_RESULTS)
    set_query_results(search_results, query_input)


def query_best_frame_embedding(query_input):
    """ Get the results of the query using the best frame embedding """
    query_embedding = generate_embeddings.generate_query_embeddings(query_input)
    search_results = opensearch.knn_query_best(query_embedding.tolist(), N_RESULTS)
    set_query_results(search_results, query_input)


def set_query_results(search_results, query_input):
    """ Set the info of the query from the search results """
    query_results = {}

    for hit in search_results['hits']['hits']:
        if hit['_source']['video_id'] not in query_results:
            query_results[hit['_source']['video_id']] = {}
        query_results[hit['_source']['video_id']][hit['_source']['annotation_id']] = {}
        query_results[hit['_source']['video_id']][hit['_source']['annotation_id']]['annotation_value'] = hit['_source'][
            'annotation_value']
        query_results[hit['_source']['video_id']][hit['_source']['annotation_id']]['start_time'] = hit['_source'][
            'start_time']
        query_results[hit['_source']['video_id']][hit['_source']['annotation_id']]['end_time'] = hit['_source'][
            'end_time']
        query_results[hit['_source']['video_id']][hit['_source']['annotation_id']]['phrase'] = hit['_source']['phrase']
        query_results[hit['_source']['video_id']][hit['_source']['annotation_id']]['similarity_score'] = hit['_score']

    print_performance_metrics(query_results, query_input)

    session['query_results'] = query_results


def query_true_expression(query_input):
    """ Get the results of the query using the ground truth """
    query_results = {}
    search_mode = session.get('search_mode', 1)
    pattern = re.compile(r'(^|\[|_|]){}($|\]|_|)'.format(query_input.lower()), re.IGNORECASE)

    for video in os.listdir(os.path.join(FRAMES_PATH, search_mode)):
        with open(os.path.join(ANNOTATIONS_PATH, f"{video}.json"), "r") as f:
            video_annotations = json.load(f)
            if search_mode in video_annotations:
                for annotation in video_annotations[search_mode]["annotations"]:
                    if annotation["value"] is not None and pattern.search(annotation["value"]):
                        if video not in query_results:
                            query_results[video] = {}
                        query_results[video][annotation["annotation_id"]] = {}
                        query_results[video][annotation["annotation_id"]]['annotation_value'] = annotation["value"]
                        query_results[video][annotation["annotation_id"]]['start_time'] = annotation["start_time"]
                        query_results[video][annotation["annotation_id"]]['end_time'] = annotation["end_time"]
                        query_results[video][annotation["annotation_id"]]['phrase'] = annotation["phrase"]
                        query_results[video][annotation["annotation_id"]]['similarity_score'] = "N/A"

    session['query_results'] = query_results

    return query_results


def print_performance_metrics(query_results, query_input):
    """ Print the performance metrics of the query
        Precision tells how many retrieved results were right
        Recall tells how many right results were retrieved
        F1 score is the harmonic mean of precision and recall """
    compare_results = query_true_expression(query_input)

    # Initialize counters for true positives, false positives and false negatives
    tp = 0  # True positives - number of annotations that were correctly retrieved
    fp = 0  # False positives - number of annotations that were retrieved but are not in the ground truth
    fn = 0  # False negatives - number of annotations that are in the ground truth but were not retrieved

    # Iterate over each video in query_results
    for video in query_results.keys():
        query_annotations = query_results[video].keys()

        # If the video is also in compare_results
        if video in compare_results:
            compare_annotations = compare_results[video].keys()

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
