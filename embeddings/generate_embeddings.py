# from sentence_embeddings import Embedder as SentenceEmbedder
from .sentence_embeddings import Embedder as SentenceEmbedder
import os
import sys
import json
import pickle

st = SentenceEmbedder()


# Generates annotation embeddings for a single video, where
# frame_annotations is a json file with several types of annotations in a video
def generate_annotation_embeddings(frame_annotations, result_dir):
    embeddings = {}

    data = json.load(frame_annotations)

    for tier in data:
        print(f"Embedding {tier} annotations")
        annotation_embeddings = {}
        if len(data[tier]['annotations']) == 0:
            continue

        # filter out the null values from the data[tier]['annotations'] list
        data[tier]['annotations'] = list(
            filter(lambda annotation: annotation['value'] is not None, data[tier]['annotations']))

        # encode only the values of each annotation
        embed_buffer = st.text_encode(list(annotation['value'] for annotation in data[tier]['annotations']))
        for i, annotation_id in enumerate(annotation['annotation_id'] for annotation in data[tier]['annotations']):
            annotation_embeddings[annotation_id] = embed_buffer[i]
        embeddings[tier] = annotation_embeddings

    video_name = os.path.splitext(os.path.basename(frame_annotations))[0]
    pickle.dump(embeddings, open(os.path.join(result_dir, video_name + '_annotation_embeddings.json.embeddings'), 'wb'))


# Generates frame embeddings for a single video
def generate_frame_embeddings(frame_dir, result_dir):

    print(f"Working on {frame_dir}")
    frame_embeddings = {}
    # iterate results
    for frame in os.listdir(frame_dir):
        full_path = os.path.abspath(os.path.join(frame_dir, frame))
        print(f"Embedding {full_path}")
        # get name of file without extension
        time = os.path.splitext(frame)[0]
        # generate embedding
        frame_embeddings[time] = st.image_encode(full_path)

    # save dict
    video_name = os.path.splitext(os.path.basename(frame_dir))[0]
    pickle.dump(frame_embeddings, open(os.path.join(result_dir, video_name + '_frame_embeddings.json.embeddings'), 'wb'))
