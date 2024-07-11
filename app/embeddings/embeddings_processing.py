import io
import json

import numpy as np
import torch

from .embeddings_generator import Embedder
import os
import pickle
import gc

from ..utils import FACIAL_EXPRESSIONS_FRAMES_DIR, EMBEDDINGS_PATH, ANNOTATIONS_PATH, FACIAL_EXPRESSIONS_ID


def generate_video_embeddings():
    """ Generates all the embeddings for a folder of video frames """

    # Initialize the Embeddings Generator using the GPU
    embedder = Embedder(check_gpu=True)

    generate_frame_embeddings(embedder)
    print("Frame embeddings generated", flush=True)

    generate_average_and_best_frame_embeddings(embedder)
    print("Average and best frame embeddings generated", flush=True)

    generate_summed_embeddings()
    print("Summed embeddings generated", flush=True)

    generate_annotations_embeddings(embedder)
    print("Annotations embeddings generated", flush=True)


def generate_frame_embeddings(eb: Embedder):
    """ Generates facial expression frame embeddings for a folder of videos, choosing 4 frames
    equally spaced for each video. """
    print("Generating frame embeddings", flush=True)

    n_embeddings = 4
    embeddings = {}
    annotations_batch_size = 32

    embeddings_file = os.path.join(EMBEDDINGS_PATH, 'frame_embeddings.json.embeddings')
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)

    for video in os.listdir(FACIAL_EXPRESSIONS_FRAMES_DIR):

        video_dir = os.path.join(FACIAL_EXPRESSIONS_FRAMES_DIR, video)

        annotations_dir = os.listdir(video_dir)

        if video in embeddings or len(annotations_dir) == 0 or video.startswith('.'):
            continue

        embeddings[video] = {}

        print(f"Working on {video}", flush=True)

        for i in range(0, len(annotations_dir), annotations_batch_size):
            annotations_batch = annotations_dir[i:i + annotations_batch_size]

            for annotation in annotations_batch:
                expression_frames_dir = os.path.join(video_dir, annotation)
                annotation_embedding = None

                all_frames = os.listdir(expression_frames_dir)

                # Select #n_embeddings frames to generate embeddings
                if len(all_frames) <= n_embeddings:
                    step_size = 1
                else:
                    # Calculate the step size
                    step_size = (len(all_frames) - 1) // n_embeddings + 1

                frames_to_encode = all_frames[::step_size]
                frames_to_encode = frames_to_encode[:n_embeddings]

                for frame in frames_to_encode:
                    full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))

                    if not os.path.isfile(full_path):
                        continue

                    # Initialize embeddings[video][annotation] as a zero tensor if it's not already initialized
                    if annotation_embedding is None:
                        annotation_embedding = torch.zeros_like(eb.image_encode(full_path))

                    # generate embedding and sum it to the total embedding
                    annotation_embedding += eb.image_encode(full_path)

                embeddings[video][annotation] = annotation_embedding

                del annotation_embedding
                torch.cuda.empty_cache()

            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)

            gc.collect()


def generate_average_and_best_frame_embeddings(eb: Embedder):
    """ Generates the average and best facial expression frame embeddings of a video for a folder of videos """
    print("Generating average and best frame embeddings", flush=True)

    average_embeddings_file = os.path.join(EMBEDDINGS_PATH, 'average_frame_embeddings.json.embeddings')
    best_embeddings_file = os.path.join(EMBEDDINGS_PATH, 'best_frame_embeddings.json.embeddings')
    annotations_batch_size = 8
    average_embeddings = {}
    best_embeddings = {}

    if os.path.exists(average_embeddings_file):
        with open(average_embeddings_file, 'rb') as f:
            average_embeddings = pickle.load(f)

    if os.path.exists(best_embeddings_file):
        with open(best_embeddings_file, 'rb') as f:
            best_embeddings = pickle.load(f)

    for video in os.listdir(FACIAL_EXPRESSIONS_FRAMES_DIR):

        video_dir = os.path.join(FACIAL_EXPRESSIONS_FRAMES_DIR, video)
        annotations_dir = os.listdir(video_dir)

        if video in average_embeddings and video in best_embeddings:
            continue

        if len(annotations_dir) == 0 or video.startswith('.'):
            continue

        average_embeddings[video] = {}
        best_embeddings[video] = {}

        print(f"Working on {video}", flush=True)

        for i in range(0, len(annotations_dir), annotations_batch_size):
            annotations_batch = annotations_dir[i:i + annotations_batch_size]

            for annotation in annotations_batch:
                expression_frames_dir = os.path.join(video_dir, annotation)
                total_embedding = None
                best_embedding = None
                best_score = -1
                frame_count = 0

                for frame in os.listdir(expression_frames_dir):
                    full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))

                    if not os.path.isfile(full_path):
                        continue

                    with torch.no_grad():  # Avoid storing computations for gradient calculation
                        current_embedding = eb.image_encode(full_path)

                    # calculate score based on the norm (or length) of the embedding vector
                    # the higher the norm, more intense and distinct the expression is, the better the embedding
                    current_score = np.linalg.norm(current_embedding.detach().cpu().numpy())

                    if current_score > best_score:
                        best_score = current_score
                        best_embedding = current_embedding

                    if total_embedding is None:
                        total_embedding = current_embedding
                    else:
                        total_embedding += current_embedding
                    frame_count += 1

                # calculate average embedding
                if frame_count > 0:
                    average_embedding = total_embedding / frame_count
                    average_embeddings[video][annotation] = average_embedding

                best_embeddings[video][annotation] = best_embedding

                del total_embedding, best_embedding
                torch.cuda.empty_cache()

            with open(average_embeddings_file, 'wb') as f:
                pickle.dump(average_embeddings, f)

            with open(best_embeddings_file, 'wb') as f:
                pickle.dump(best_embeddings, f)

            gc.collect()


def generate_summed_embeddings():
    """ Generate the summed embeddings for all the facial expressions annotations' values """
    print("Generating summed embeddings", flush=True)

    # Load embeddings
    with open(os.path.join(EMBEDDINGS_PATH, 'frame_embeddings.json.embeddings'), 'rb') as f:
        base_embeddings = pickle.load(f)
    with open(os.path.join(EMBEDDINGS_PATH, 'average_frame_embeddings.json.embeddings'), 'rb') as f:
        average_embeddings = pickle.load(f)
    with open(os.path.join(EMBEDDINGS_PATH, 'best_frame_embeddings.json.embeddings'), 'rb') as f:
        best_embeddings = pickle.load(f)

    summed_embeddings_file = os.path.join(EMBEDDINGS_PATH, 'summed_frame_embeddings.json.embeddings')

    summed_embeddings = {}
    if os.path.exists(summed_embeddings_file):
        with open(summed_embeddings_file, 'rb') as f:
            summed_embeddings = pickle.load(f)

    for video, annotations in base_embeddings.items():

        if video in summed_embeddings or video not in base_embeddings:
            continue

        summed_embeddings[video] = {}

        print(f"Working on {video}", flush=True)

        for annotation_id, base_emb in annotations.items():
            # Ensure the annotation exists in all embeddings
            if video in average_embeddings and annotation_id in average_embeddings[video] and \
                    video in best_embeddings and annotation_id in best_embeddings[video]:
                # Sum the embeddings
                summed_emb = base_emb + average_embeddings[video][annotation_id] + best_embeddings[video][annotation_id]
                summed_embeddings[video][annotation_id] = summed_emb.cpu()

    # Save the summed embeddings
    with open(summed_embeddings_file, 'wb') as f:
        pickle.dump(summed_embeddings, f)


def generate_annotations_embeddings(eb: Embedder):
    """ Generate the embeddings for all the facial expressions annotations' values """
    print("Generating annotations embeddings", flush=True)

    embeddings = {}

    embeddings_file = os.path.join(EMBEDDINGS_PATH, 'annotations_embeddings.json.embeddings')
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)

    for video_annotations in os.listdir(ANNOTATIONS_PATH):

        video_name = video_annotations.split(".")[0]

        if video_name in embeddings or video_annotations.startswith('.'):
            continue

        print(f"Working on {video_annotations}", flush=True)

        annotation_json = os.path.join(ANNOTATIONS_PATH, video_annotations)

        embeddings[video_name] = {}

        with open(annotation_json, 'r') as f:
            annotations = json.load(f)

            if FACIAL_EXPRESSIONS_ID not in annotations:
                continue

            expressions = annotations[FACIAL_EXPRESSIONS_ID]["annotations"]

            for expression in expressions:
                annotation_id = expression["annotation_id"]
                annotation_value = expression["value"]

                if annotation_value is not None:
                    embeddings[video_name][annotation_id] = eb.text_encode(annotation_value.lower())
                else:
                    embeddings[video_name][annotation_id] = torch.zeros(512)

    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)


def generate_query_embeddings(query_input, eb: Embedder):
    """ Generate the user queries embeddings """
    return eb.text_encode(query_input.lower())
