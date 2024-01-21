# from sentence_embeddings import Embedder as SentenceEmbedder
from sentence_embeddings import Embedder as SentenceEmbedder
import os
import sys
import json
import pickle

st = SentenceEmbedder()


# Generates annotation embeddings, where frame_annotations is the path
# to the json files of several types of annotations of multiple videos
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


# Generates frame embeddings for a folder of videos
def generate_frame_embeddings(frames_dir, result_dir):
    embeddings = {}

    for video in os.listdir(frames_dir):
        print(f"Working on {video}")
        video_dir = os.path.join(frames_dir, video)
        embeddings[video] = {}
        for frame in os.listdir(video_dir):
            print(f"Embedding {frame}")
            full_path = os.path.abspath(os.path.join(video_dir, frame))
            # get name of file without extension
            time = os.path.splitext(frame)[0]
            # generate embedding
            embeddings[video][time] = st.image_encode(full_path)

    pickle.dump(embeddings, open(os.path.join(result_dir, 'frame_embeddings.json.embeddings'), 'wb'))
