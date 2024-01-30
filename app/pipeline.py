import json
import os
import pickle
import torch
import io
import time

from .eaf_parser import eaf_parser
from .opensearch.opensearch import LGPOpenSearch
from .frame_extraction import frame_extraction
from .embeddings import generate_embeddings

RESULTS_PATH = "app/static/videofiles"  # sys.argv[exp9]
VIDEO_PATH = "app/static/videofiles/mp4"  # sys.argv[exp10]
EAF_PATH = "app/static/videofiles/eaf"  # sys.argv[exp11]
ANNOTATIONS_PATH = "app/static/videofiles/annotations"  # sys.argv[4]
FRAMES_PATH = "app/static/videofiles/frames"  # sys.argv[5]
EMBEDDINGS_PATH = "app/embeddings"  # sys.argv[6]

PHRASES_ID = "LP_P1 transcrição livre"
FACIAL_EXPRESSIONS_ID = "GLOSA_P1_EXPRESSAO"

opensearch = LGPOpenSearch()


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def gen_doc(video_id: str, annotation_id: str, annotation_value: str,
            base_frame_embedding, average_frame_embedding, best_frame_embedding):
    return {
        "video_id": video_id,
        "annotation_id": annotation_id,
        "annotation_value": annotation_value,
        "base_frame_embedding": base_frame_embedding,
        "average_frame_embedding": average_frame_embedding,
        "best_frame_embedding": best_frame_embedding
    }


def preprocess_videos():
    """ Preprocess videos, generate embeddings and index them in OpenSearch."""

    # Generate json file for videos with annotations and timestamps
    eaf_parser.parse_eaf_files(EAF_PATH)
    print("Annotations generated")

    # Extract facial expressions frames
    frame_extraction.extract_frames(VIDEO_PATH, FRAMES_PATH, ANNOTATIONS_PATH)
    print("Extracted facial expressions frames")

    facial_expressions_frames_path = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID)

    # Generate the base frames embeddings
    generate_embeddings.generate_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH)
    print("Frame embeddings generated")

    # Generate the average frames embeddings
    generate_embeddings.generate_average_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH)
    print("Average frame embeddings generated")

    # Generate the best frame embeddings
    generate_embeddings.generate_best_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH)
    print("Best frame embeddings generated")

    print("Reading base frames embeddings")
    with open(os.path.join(EMBEDDINGS_PATH, "frame_embeddings.json.embeddings"), "rb") as f:
        base_frame_embeddings = CPU_Unpickler(f).load()

    print("Reading average frame embeddings")
    with open(os.path.join(EMBEDDINGS_PATH, "average_frame_embeddings.json.embeddings"), "rb") as f:
        average_frame_embeddings = CPU_Unpickler(f).load()

    print("Reading best frame embeddings")
    with open(os.path.join(EMBEDDINGS_PATH, "best_frame_embeddings.json.embeddings"), "rb") as f:
        best_frame_embeddings = CPU_Unpickler(f).load()

    print("ENTERING INDEXING LOOP")
    opensearch.create_index()
    for video_id in os.listdir(facial_expressions_frames_path):

        # Read annotations
        with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
            video_annotations = json.load(f)

            if FACIAL_EXPRESSIONS_ID not in video_annotations:
                continue

            annotations = video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]

            for annotation in annotations:
                annotation_id = annotation["annotation_id"]
                annotation_value = annotation["value"]


                doc = gen_doc(
                    video_id=video_id,
                    annotation_id=annotation_id,
                    annotation_value=annotation_value,
                    base_frame_embedding=base_frame_embeddings[video_id][annotation_id].tolist(),
                    average_frame_embedding=average_frame_embeddings[video_id][annotation_id].tolist(),
                    best_frame_embedding=best_frame_embeddings[video_id][annotation_id].tolist()
                )

                opensearch.index_if_not_exists(doc)