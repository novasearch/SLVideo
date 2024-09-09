import datetime
import json
import os
import re
from collections import OrderedDict
from datetime import datetime as dt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

from flask import (
    Blueprint, flash, redirect, render_template, request, session, url_for
)
from sklearn.preprocessing import StandardScaler

from .embeddings import embeddings_processing
from .utils import embedder, opensearch, CPU_Unpickler, FRAMES_PATH, FACIAL_EXPRESSIONS_ID, PHRASES_ID, \
    ANNOTATIONS_PATH, AVERAGE_FRAMES_EMBEDDINGS_FILE, BASE_FRAMES_EMBEDDINGS_FILE, BEST_FRAMES_EMBEDDINGS_FILE, \
    SUMMED_FRAMES_EMBEDDINGS_FILE, ALL_FRAMES_EMBEDDINGS_FILE

N_RESULTS = 10
N_FRAMES_TO_DISPLAY = 6

bp = Blueprint('query', __name__)


@bp.route("/query", methods=("GET", "POST"))
def query():
    """Query for a video"""
    if request.method == "POST":
        query_input = request.form["query"]
        session['query_input'] = query_input
        selected_field = int(request.form.get('field'))
        session['selected_field'] = selected_field
        session['search_mode'] = request.form.get('mode')
        error = None

        if not query_input:
            error = "Query is required."

        if error is not None:
            flash(error)
        else:

            print()
            print("-------------------------------------")
            if selected_field == 1:  # Base Frames Embeddings
                print(f"Searching for '{query_input}' using Base Frames Embeddings...")
                session['query_results'] = query_frames_embeddings(query_input)
            elif selected_field == 2:  # Average Frames Embeddings
                print(f"Searching for '{query_input}' using Average Frames Embeddings...")
                session['query_results'] = query_average_frames_embeddings(query_input)
            elif selected_field == 3:  # Best Frame Embeddings
                print(f"Searching for '{query_input}' using Best Frame Embeddings...")
                session['query_results'] = query_best_frame_embedding(query_input)
            elif selected_field == 4:  # Summed Frames Embeddings
                print(f"Searching for '{query_input}' using Summed Frames Embeddings...")
                session['query_results'] = query_summed_frames_embeddings(query_input)
            elif selected_field == 5:  # All Frames Embeddings
                print(f"Searching for '{query_input}' using All Frames Embeddings...")
                session['query_results'] = query_all_frames_embeddings(query_input)
            elif selected_field == 6:  # Combined Frames Embeddings
                print(f"Searching for '{query_input}' using Combined Frames Embeddings...")
                session['query_results'] = query_combined_frames_embeddings(query_input)
            elif selected_field == 7:  # Annotations Embeddings
                print(f"Searching for '{query_input}' using Annotations Embeddings...")
                session['query_results'] = query_annotations_embeddings(query_input)
            else:  # True Expression / Ground Truth
                session['query_results'] = query_true_expression(query_input)

            print()

            # If there are no results, display a message
            if query_input is not None and not session['query_results']:
                flash("No results found.")
                return render_template("query/query.html")

            return redirect(url_for("query.videos_results"))

    return render_template("query/query.html")


@bp.route("/videos_results", methods=("GET", "POST"))
def videos_results():
    """ Display the videos that contain the query results """
    query_results = session.get('query_results')

    search_mode = session.get('search_mode', 1)

    # Sort the query results by the number of annotations
    query_results = OrderedDict(sorted(query_results.items(), key=lambda x: len(x[1]), reverse=True))

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

    return render_template("query/videos_results.html", frames=frames, videos_info=videos_info,
                           search_mode=search_mode)


@bp.route("/clips_results/<video>", methods=("GET", "POST"))
def clips_results(video):
    """ Display the video segments of one video that contain the query results """
    query_results = session.get('query_results')[video]
    query_input = session.get('query_input')
    search_mode = session.get('search_mode', 1)

    # Sort the query results by similarity score
    query_results = OrderedDict(sorted(query_results.items(), key=lambda x: x[1]['similarity_score'], reverse=True))

    frames = {}

    # Collect information about the retrieved video segments
    for annotation_id in query_results:
        frames_path = os.path.join(FRAMES_PATH, search_mode, video, annotation_id)
        frames[annotation_id] = os.listdir(frames_path)

        converted_start_time = str(datetime.timedelta(seconds=int(query_results[annotation_id]["start_time"]) // 1000))
        converted_end_time = str(datetime.timedelta(seconds=int(query_results[annotation_id]["end_time"]) // 1000))

        query_results[annotation_id]["converted_start_time"] = converted_start_time
        query_results[annotation_id]["converted_end_time"] = converted_end_time

    if search_mode == FACIAL_EXPRESSIONS_ID:
        # Not all frames of each expression are going to be displayed
        frames_to_display = get_frames_to_display(frames)

        if request.method == "POST":
            button_clicked = request.form.get("button_clicked")
            selected_annotation = request.form.get("selected_annotation")

            if button_clicked == 'edit':
                return redirect(
                    url_for("annotations.edit_annotation", video_id=video, annotation_id=selected_annotation))
            elif button_clicked == 'thesaurus':
                return redirect(url_for("query.thesaurus_results", video_id=video, annotation_id=selected_annotation))
        return render_template("query/clips_results/expressions_clips.html", frames=frames_to_display,
                               frames_info=query_results, query_input=query_input,
                               search_mode=search_mode, video=video)
    elif search_mode == PHRASES_ID:

        # Display all frames of each phrase
        frames_to_display = frames

        return render_template("query/clips_results/phrases_clips.html", frames=frames_to_display,
                               frames_info=query_results, query_input=query_input,
                               search_mode=search_mode, video=video)


@bp.route("/thesaurus/<video_id>/<annotation_id>", methods=("GET", "POST"))
def thesaurus_results(video_id, annotation_id):
    """ Display the thesaurus results for a similar sign """
    search_results = query_thesaurus(video_id, annotation_id)
    search_mode = session.get('search_mode', 1)

    frames = {}

    # Collect information about the retrieved video segments
    for video in search_results:
        for annotation_id in search_results[video]:
            frames_path = os.path.join(FRAMES_PATH, search_mode, video, annotation_id)

            frames[video + "_" + annotation_id] = os.listdir(frames_path)

            converted_start_time = str(
                datetime.timedelta(seconds=int(search_results[video][annotation_id]["start_time"]) // 1000))
            converted_end_time = str(
                datetime.timedelta(seconds=int(search_results[video][annotation_id]["end_time"]) // 1000))

            search_results[video][annotation_id]["converted_start_time"] = converted_start_time
            search_results[video][annotation_id]["converted_end_time"] = converted_end_time

    # Not all frames of each expression are going to be displayed
    frames_to_display = get_frames_to_display(frames, n_frames=1)

    # Initialize a dictionary to store the frames information so that the keys
    # are a composition of the video_id and the annotation_id
    frames_info = {}
    for video in search_results:
        for annotation_id in search_results[video]:
            frames_info[video + "_" + annotation_id] = search_results[video][annotation_id]
            frames_info[video + "_" + annotation_id]["annotation_id"] = annotation_id

    return render_template("query/clips_results/thesaurus_clips.html", frames=frames_to_display,
                           frames_info=frames_info, search_mode=search_mode, video=video_id)


def query_frames_embeddings(query_input):
    """ Get the results of the query using the frames embeddings """
    query_embedding = embeddings_processing.generate_query_embeddings(query_input, embedder)
    search_results = opensearch.knn_query(query_embedding.tolist(), N_RESULTS)
    return set_query_results(search_results, query_input)


def query_average_frames_embeddings(query_input):
    """ Get the results of the query using the average of the frames embeddings """
    query_embedding = embeddings_processing.generate_query_embeddings(query_input, embedder)
    search_results = opensearch.knn_query_average(query_embedding.tolist(), N_RESULTS)
    return set_query_results(search_results, query_input)


def query_best_frame_embedding(query_input):
    """ Get the results of the query using the best frame embedding """
    query_embedding = embeddings_processing.generate_query_embeddings(query_input, embedder)
    search_results = opensearch.knn_query_best(query_embedding.tolist(), N_RESULTS)
    return set_query_results(search_results, query_input)


def query_summed_frames_embeddings(query_input):
    """ Get the results of the query using the summed frames embeddings """
    query_embedding = embeddings_processing.generate_query_embeddings(query_input, embedder)
    search_results = opensearch.knn_query_summed(query_embedding.tolist(), N_RESULTS)
    return set_query_results(search_results, query_input)


def query_all_frames_embeddings(query_input):
    """ Get the results of the query using all frames embeddings """
    query_embedding = embeddings_processing.generate_query_embeddings(query_input, embedder)
    search_results = opensearch.knn_query_all(query_embedding.tolist(), N_RESULTS)
    return set_query_results(search_results, query_input)


def query_combined_frames_embeddings(query_input):
    """ Get the results of the query using the combined frames embeddings """
    query_embedding = embeddings_processing.generate_query_embeddings(query_input, embedder)
    search_results = opensearch.knn_query_combined(query_embedding.tolist(), N_RESULTS)
    return set_query_results(search_results, query_input)


def query_annotations_embeddings(query_input):
    """ Get the results of the query using the annotations embeddings """
    query_embedding = embeddings_processing.generate_query_embeddings(query_input, embedder)
    search_results = opensearch.knn_query_annotations(query_embedding.tolist(), N_RESULTS)
    return set_query_results(search_results, query_input)


def query_thesaurus(video_id, annotation_id):
    """ Get the results of querying for a similar sign """

    selected_field = session.get('selected_field')
    if selected_field == 1:  # Base Frames Embeddings
        with open(BASE_FRAMES_EMBEDDINGS_FILE, "rb") as f:
            base_frame_embeddings = CPU_Unpickler(f).load()
        embedding = base_frame_embeddings[video_id][annotation_id].tolist()
        search_results = opensearch.knn_query(embedding, N_RESULTS)
        print(f"Searching using Base Frames Embeddings...")
    elif selected_field == 2:  # Average Frames Embeddings
        with open(AVERAGE_FRAMES_EMBEDDINGS_FILE, "rb") as f:
            average_frame_embeddings = CPU_Unpickler(f).load()
        embedding = average_frame_embeddings[video_id][annotation_id].tolist()
        search_results = opensearch.knn_query_average(embedding, N_RESULTS)
        print(f"Searching using Average Frames Embeddings...")
    elif selected_field == 3:  # Best Frame Embeddings
        with open(BEST_FRAMES_EMBEDDINGS_FILE, "rb") as f:
            best_frame_embeddings = CPU_Unpickler(f).load()
        embedding = best_frame_embeddings[video_id][annotation_id].tolist()
        search_results = opensearch.knn_query_best(embedding, N_RESULTS)
        print(f"Searching using Best Frame Embeddings...")
    elif selected_field == 4:  # Summed Frames Embeddings
        with open(SUMMED_FRAMES_EMBEDDINGS_FILE, "rb") as f:
            summed_frame_embeddings = CPU_Unpickler(f).load()
        embedding = summed_frame_embeddings[video_id][annotation_id].tolist()
        search_results = opensearch.knn_query_summed(embedding, N_RESULTS)
        print(f"Searching using Summed Frames Embeddings...")
    elif selected_field == 5:  # All Frames Embeddings
        with open(ALL_FRAMES_EMBEDDINGS_FILE, "rb") as f:
            all_frame_embeddings = CPU_Unpickler(f).load()
        embedding = all_frame_embeddings[video_id][annotation_id].tolist()
        search_results = opensearch.knn_query_all(embedding, N_RESULTS)
        print(f"Searching using All Frames Embeddings...")
    else :  # Combined Frames Embeddings or Annotations Embeddings or True Expression
        with open(AVERAGE_FRAMES_EMBEDDINGS_FILE, "rb") as f:
            average_frame_embeddings = CPU_Unpickler(f).load()
        embedding = average_frame_embeddings[video_id][annotation_id].tolist()
        search_results = opensearch.knn_query_combined(embedding, N_RESULTS)

    annotation_value = None
    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)
        for annotation in video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]:
            if annotation["annotation_id"] == annotation_id:
                annotation_value = annotation['value']
                break
    return set_query_results(search_results, annotation_value, plot_tsne=True)


def set_query_results(search_results, query_input=None, plot_tsne=False):
    """ Set the info of the query from the search results """
    query_results = {}

    videos_annotations = {}

    for hit in search_results['hits']['hits']:
        video_id = hit['_source']['video_id']
        annotation_id = hit['_source']['annotation_id']

        if video_id not in query_results:
            query_results[video_id] = {}
            with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
                videos_annotations[video_id] = json.load(f)

        query_results[video_id][annotation_id] = {}
        query_results[video_id][annotation_id]['video_id'] = video_id
        query_results[video_id][annotation_id]['similarity_score'] = hit['_score']

        for annotation in videos_annotations[video_id][FACIAL_EXPRESSIONS_ID]["annotations"]:
            if annotation["annotation_id"] == annotation_id:
                query_results[video_id][annotation_id]['annotation_value'] = annotation['value']
                query_results[video_id][annotation_id]['start_time'] = annotation['start_time']
                query_results[video_id][annotation_id]['end_time'] = annotation['end_time']
                query_results[video_id][annotation_id]['phrase'] = annotation['phrase']
                query_results[video_id][annotation_id]['user_rating'] = annotation['user_rating']
                break

    if query_input:
        print_performance_metrics(query_results, query_input)

    # if the query_embedding is given, gets the coordinates of the query and the results in 2D
    if plot_tsne:
        coordinates = get_results_tsne(
            np.array([hit['_source']['average_frame_embedding'] for hit in search_results['hits']['hits']]))
        for video_id in query_results.keys():
            for i, annotation_id in enumerate(query_results[video_id].keys()):
                query_results[video_id][annotation_id]['coordinates'] = coordinates[i].tolist()

    return query_results


def query_true_expression(query_input):
    """ Get the results of the query using the ground truth """
    query_results = {}
    search_mode = session.get('search_mode', 1)
    escaped_query_input = re.escape(query_input.lower())
    pattern = re.compile(r'(^|\[|_|]|\(|-|\s){}($|\]|_|(?=\W)|\)|-|\s)'.format(escaped_query_input), re.IGNORECASE)
    for video in os.listdir(os.path.join(FRAMES_PATH, search_mode)):
        if video.startswith("."):
            continue
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
                        if search_mode == FACIAL_EXPRESSIONS_ID:
                            query_results[video][annotation["annotation_id"]]['phrase'] = annotation["phrase"]
                        query_results[video][annotation["annotation_id"]]['similarity_score'] = "N/A"

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

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("-------------------------------------")


def get_frames_to_display(frames, n_frames=N_FRAMES_TO_DISPLAY):
    """ Get the frames to display """
    frames_to_display = {}

    # Display only #num_frames_to_display frames of each expression
    for expression, all_frames in frames.items():

        # Calculate the step size
        if len(all_frames) <= n_frames:
            step_size = 1
        else:
            step_size = (len(all_frames) - 1) // n_frames + 1

        # Select frames to display
        frames_to_display[expression] = all_frames[::step_size]

        # Ensure that only num_frames_to_display frames are selected
        frames_to_display[expression] = frames_to_display[expression][:n_frames]

    return frames_to_display


def get_results_tsne(results_embeddings):
    """ Get the 2D coordinates of the results using t-SNE """
    # Standardize the embeddings
    scaler = StandardScaler()
    results_embeddings = scaler.fit_transform(results_embeddings)

    # Reduce dimensionality with PCA before t-SNE for faster processing
    n_pca_components = min(len(results_embeddings), N_RESULTS,
                           50)  # Ensure PCA components do not exceed available samples
    pca = PCA(n_components=n_pca_components)
    results_embeddings_reduced = pca.fit_transform(results_embeddings)

    # Dynamically adjust perplexity based on the number of results
    perplexity_value = min(max(5, N_RESULTS // 2), len(results_embeddings) - 1)

    # Use t-SNE to reduce the dimensionality of the embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity_value)
    embeddings_2d = tsne.fit_transform(results_embeddings_reduced)

    return embeddings_2d


@bp.route("/update_annotation_rating", methods=["POST"])
def update_annotation_rating():
    """ Update the rating of an annotation in the session variable """
    data = request.get_json()
    video_id = data['video_id']
    annotation_id = data['annotation_id']
    rating = data['rating']

    query_results = session.get('query_results', {})

    query_results[video_id][annotation_id]['user_rating'] = rating

    session['query_results'] = query_results

    return '', 204


@bp.route('/update_annotation_info', methods=["POST"])
def update_annotation_info():
    """ Update the information of an annotation in the session variable """
    data = request.get_json()
    video_id = data['video_id']
    annotation_id = data['annotation_id']

    query_results = session.get('query_results', {})

    if video_id not in query_results or annotation_id not in query_results[video_id]:
        return '', 204

    action = data['action_type']

    if action == 'delete':
        del query_results[video_id][annotation_id]
        session['query_results'] = query_results
        return '', 204
    elif action == 'edit':
        annotation_value = data['expression']
        phrase = data['phrase']
        start_time = convert_to_milliseconds(data['start_time'])
        end_time = convert_to_milliseconds(data['end_time'])

        query_results[video_id][annotation_id]['annotation_value'] = annotation_value
        query_results[video_id][annotation_id]['phrase'] = phrase
        query_results[video_id][annotation_id]['start_time'] = start_time
        query_results[video_id][annotation_id]['end_time'] = end_time

        session['query_results'] = query_results

        return '', 204


def convert_to_milliseconds(time_str):
    # Convert the time string to a datetime object
    time_obj = dt.strptime(time_str, '%H:%M:%S')

    # Calculate the total milliseconds
    milliseconds = (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000

    return milliseconds
