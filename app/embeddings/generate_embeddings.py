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

# Generates annotation embeddings, where frame_annotations is the path
# to the json files of several types of annotations of multiple videos
#
# ** Not being used currently **
def generate_annotation_embeddings(annotations_dir, result_dir):
    embeddings = {}

    for annotations_file in os.listdir(annotations_dir):

        with open(os.path.join(annotations_dir, annotations_file), 'r') as f:

            print(f"Embedding {annotations_file} annotations")
            data = json.load(f)

            for tier in data:
                annotation_embeddings = {}
                if len(data[tier]['annotations']) == 0:
                    continue

                # filter out the null values from the data[tier]['annotations'] list
                data[tier]['annotations'] = list(
                    filter(lambda annotation: annotation['value'] is not None, data[tier]['annotations']))

                # encode only the values of each annotation
                embed_buffer = st.text_encode(list(annotation['value'] for annotation in data[tier]['annotations']))
                for i, annotation_id in enumerate(
                        annotation['annotation_id'] for annotation in data[tier]['annotations']):
                    annotation_embeddings[annotation_id] = embed_buffer[i]

                embeddings[annotations_file][tier] = annotation_embeddings

    pickle.dump(embeddings, open(os.path.join(result_dir, 'annotation_embeddings.json.embeddings'), 'wb'))


# Generates frame embeddings for all the frames of a folder of videos
def generate_frame_embeddings(frames_dir, result_dir):
    embeddings_file = os.path.join(result_dir, 'frame_embeddings.json.embeddings')
    n_embeddings = 4

    # Load existing embeddings if file exists
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = CPU_Unpickler(f).load()
    else:
        embeddings = {}

    for video in os.listdir(frames_dir):
        # Skip video if its embeddings already exist
        if video in embeddings:
            continue

        print(f"Working on {video}")
        video_dir = os.path.join(frames_dir, video)
        embeddings[video] = {}

        for annotation in os.listdir(video_dir):
            expression_frames_dir = os.path.join(video_dir, annotation)
            embeddings[video][annotation] = {}

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
                frame_id = os.path.splitext(frame)[0]
                # generate embedding
                embeddings[video][annotation][frame_id] = st.image_encode(full_path)

    pickle.dump(embeddings, open(os.path.join(result_dir, 'frame_embeddings.json.embeddings'), 'wb'))


# Generates the average of each facial expression frames embeddings of a video for a folder of videos
def generate_average_frame_embeddings(frames_dir, result_dir):
    embeddings_file = os.path.join(result_dir, 'average_frame_embeddings.json.embeddings')

    # Load existing embeddings if file exists
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = CPU_Unpickler(f).load()
    else:
        embeddings = {}

    for video in os.listdir(frames_dir):
        # Skip video if its embeddings already exist
        if video in embeddings:
            continue

        print(f"Working on {video}")
        video_dir = os.path.join(frames_dir, video)
        embeddings[video] = {}

        for annotation in os.listdir(video_dir):
            expression_frames_dir = os.path.join(video_dir, annotation)
            embeddings[video][annotation] = {}
            total_embedding = None
            frame_count = 0

            for frame in os.listdir(expression_frames_dir):
                full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))
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
                embeddings[video][annotation] = average_embedding

    pickle.dump(embeddings, open(os.path.join(result_dir, 'average_frame_embeddings.json.embeddings'), 'wb'))


# Generates only the best facial expression frame embedding of a video for a folder of videos
def generate_best_frame_embeddings(frames_dir, result_dir):
    embeddings_file = os.path.join(result_dir, 'best_frame_embeddings.json.embeddings')

    # Load existing embeddings if file exists
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = CPU_Unpickler(f).load()
    else:
        embeddings = {}

    for video in os.listdir(frames_dir):
        # Skip video if its embeddings already exist
        if video in embeddings:
            continue

        print(f"Working on {video}")
        video_dir = os.path.join(frames_dir, video)
        embeddings[video] = {}

        for annotation in os.listdir(video_dir):
            expression_frames_dir = os.path.join(video_dir, annotation)
            embeddings[video][annotation] = {}
            best_embedding = None
            best_score = -1

            for frame in os.listdir(expression_frames_dir):
                full_path = os.path.abspath(os.path.join(expression_frames_dir, frame))
                # generate embedding
                current_embedding = st.image_encode(full_path)

                # calculate score based on the norm (or length) of the embedding vector
                # the higher the norm, more intense and disctinct the expression is, the better the embedding
                current_score = np.linalg.norm(current_embedding)

                if current_score > best_score:
                    best_score = current_score
                    best_embedding = current_embedding

            embeddings[video][annotation] = best_embedding

    pickle.dump(embeddings, open(os.path.join(result_dir, 'best_frame_embeddings.json.embeddings'), 'wb'))
