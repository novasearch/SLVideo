# from sentence_embeddings import Embedder as SentenceEmbedder
import io

import numpy as np
import torch

from .sentence_embeddings import Embedder as SentenceEmbedder
import os
import json
import pickle

st = SentenceEmbedder()


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def generate_frame_embeddings(frames_dir, result_dir):
    """Generates frame embeddings for all the frames of a folder of videos"""
    print("Generating frame embeddings")
    n_embeddings = 4

    for video in os.listdir(frames_dir):
        embeddings_file = os.path.join(result_dir, video + '_frame_embeddings.json.embeddings')

        if os.path.exists(embeddings_file):
            continue

        print(f"Working on {video}")

        video_dir = os.path.join(frames_dir, video)
        embeddings = {}

        for annotation in os.listdir(video_dir):
            print(f"Processing annotation {annotation}")
            expression_frames_dir = os.path.join(video_dir, annotation)
            embeddings[annotation] = None

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

                # Skip if the path is not a file
                if not os.path.isfile(full_path):
                    continue

                # Initialize embeddings[video][annotation] as a zero tensor if it's not already initialized
                if embeddings[annotation] is None:
                    embeddings[annotation] = torch.zeros_like(st.image_encode(full_path))

                # generate embedding and sum it to the total embedding
                embeddings[annotation] += st.image_encode(full_path)

        pickle.dump(embeddings, open(embeddings_file, 'wb'))


# Generates the average of each facial expression frames embeddings of a video for a folder of videos
def generate_average_frame_embeddings(frames_dir, result_dir):
    print("Generating average frame embeddings")
    for video in os.listdir(frames_dir):
        embeddings_file = os.path.join(result_dir, video + '_average_frame_embeddings.json.embeddings')

        if os.path.exists(embeddings_file):
            continue

        print(f"Working on {video}")
        video_dir = os.path.join(frames_dir, video)
        embeddings = {}

        for annotation in os.listdir(video_dir):
            expression_frames_dir = os.path.join(video_dir, annotation)
            embeddings[annotation] = {}
            total_embedding = None
            frame_count = 0

            for frame in os.listdir(expression_frames_dir):
                full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))

                # Skip if the path is not a file
                if not os.path.isfile(full_path):
                    continue

                # generate embedding
                current_embedding = st.image_encode(full_path)
                if total_embedding is None:
                    total_embedding = current_embedding
                else:
                    total_embedding += current_embedding
                frame_count += 1

            # calculate average embedding
            if frame_count > 0:  # avoid division by zero
                average_embedding = total_embedding / frame_count
                embeddings[annotation] = average_embedding

        pickle.dump(embeddings, open(embeddings_file, 'wb'))


# Generates only the best facial expression frame embedding of a video for a folder of videos
def generate_best_frame_embeddings(frames_dir, result_dir):
    print("Generating best frame embeddings")

    for video in os.listdir(frames_dir):
        embeddings_file = os.path.join(result_dir, video + '_best_frame_embeddings.json.embeddings')

        if os.path.exists(embeddings_file):
            continue

        # Skip video if its embeddings already exist
        if video in embeddings:
            continue

        print(f"Working on {video}")
        video_dir = os.path.join(frames_dir, video)
        embeddings = {}

        for annotation in os.listdir(video_dir):
            expression_frames_dir = os.path.join(video_dir, annotation)
            embeddings[annotation] = {}
            best_embedding = None
            best_score = -1

            for frame in os.listdir(expression_frames_dir):
                full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))

                # Skip if the path is not a file
                if not os.path.isfile(full_path):
                    continue

                # generate embedding
                current_embedding = st.image_encode(full_path)

                # calculate score based on the norm (or length) of the embedding vector
                # the higher the norm, more intense and disctinct the expression is, the better the embedding
                current_score = np.linalg.norm(current_embedding)

                if current_score > best_score:
                    best_score = current_score
                    best_embedding = current_embedding

            embeddings[annotation] = best_embedding

        pickle.dump(embeddings, open(embeddings_file, 'wb'))


# Generate query embeddings
def generate_query_embeddings(query_input):
    return st.text_encode(query_input)


def generate_annotation_frames_embeddings(video_dir, annotation_id):
    n_embeddings = 4

    expression_frames_dir = os.path.join(video_dir, annotation_id)
    total_embeddings = None

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

        # Skip if the path is not a file
        if not os.path.isfile(full_path):
            continue

        # Initialize embeddings[annotation] as a zero tensor if it's not already initialized
        if total_embeddings is None:
            total_embeddings = torch.zeros_like(st.image_encode(full_path))

        # generate embedding and sum it to the total embedding
        total_embeddings += st.image_encode(full_path)

    return total_embeddings


def generate_annotation_average_frames_embeddings(video_dir, annotation_id):
    expression_frames_dir = os.path.join(video_dir, annotation_id)
    total_embedding = None
    frame_count = 0

    for frame in os.listdir(expression_frames_dir):
        full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))

        # Skip if the path is not a file
        if not os.path.isfile(full_path):
            continue

        # generate embedding
        current_embedding = st.image_encode(full_path)
        if total_embedding is None:
            total_embedding = current_embedding
        else:
            total_embedding += current_embedding
        frame_count += 1

    # calculate average embedding
    if frame_count > 0:  # avoid division by zero
        average_embedding = total_embedding / frame_count

    return average_embedding


def generate_annotation_best_frame_embeddings(video_dir, annotation_id):
    expression_frames_dir = os.path.join(video_dir, annotation_id)
    best_embedding = None
    best_score = -1

    for frame in os.listdir(expression_frames_dir):
        full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))

        # Skip if the path is not a file
        if not os.path.isfile(full_path):
            continue

        # generate embedding
        current_embedding = st.image_encode(full_path)

        # calculate score based on the norm (or length) of the embedding vector
        # the higher the norm, more intense and disctinct the expression is, the better the embedding
        current_score = np.linalg.norm(current_embedding)

        if current_score > best_score:
            best_score = current_score
            best_embedding = current_embedding

    return best_embedding
