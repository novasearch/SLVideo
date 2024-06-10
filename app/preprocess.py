import json
import os
import pickle
import subprocess

import torch
import io

from .eaf_parser import eaf_parser
from .opensearch.opensearch import LGPOpenSearch
from .embeddings import embeddings_processing

# Define the paths for the results, videos, eaf files, annotations, frames, and embeddings
RESULTS_PATH = "app/static/videofiles"  # sys.argv[exp9]
VIDEO_PATH = "app/static/videofiles/mp4"  # sys.argv[exp10]
EAF_PATH = "app/static/videofiles/eaf"  # sys.argv[exp11]
ANNOTATIONS_PATH = "app/static/videofiles/annotations"  # sys.argv[4]
FRAMES_PATH = "app/static/videofiles/frames"  # sys.argv[5]
EMBEDDINGS_PATH = "app/embeddings"  # sys.argv[6]

PHRASES_ID = "LP_P1 transcrição livre"
FACIAL_EXPRESSIONS_ID = "GLOSA_P1_EXPRESSAO"

# Initialize the OpenSearch client
opensearch = LGPOpenSearch()

# Initialize the Embeddings Generator
embedder = embeddings_processing.Embedder(check_gpu=True)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def run_in_env(script_path, env_path):
    """ Run a script in a virtual environment """
    activate_script = f'{env_path}\\Scripts\\activate' if os.name == 'nt' else f'source {env_path}/bin/activate'
    print(f"Activating environment: {activate_script}", flush=True)
    command = f"{activate_script} && python {script_path} && deactivate"
    process = subprocess.Popen(command, shell=True, executable="/bin/bash", stderr=subprocess.PIPE)
    _, err = process.communicate()
    if process.returncode != 0:
        print(f"Error occurred: {err.decode()}")
    process.wait()


def gen_doc(video_id: str, annotation_id: str, annotation_value: str,
            base_frame_embedding, average_frame_embedding, best_frame_embedding,
            start_time: int, end_time: int, phrase: str):
    """ Generate a document for indexing in OpenSearch """
    return {
        "video_id": video_id,
        "annotation_id": annotation_id,
        "annotation_value": annotation_value,
        "base_frame_embedding": base_frame_embedding,
        "average_frame_embedding": average_frame_embedding,
        "best_frame_embedding": best_frame_embedding,
        "start_time": start_time,
        "end_time": end_time,
        "phrase": phrase
    }


""" Preprocess videos, extract frames, generate embeddings and create indexes in OpenSearch """

# Generate json file for videos with annotations and timestamps
eaf_parser.parse_eaf_files(EAF_PATH)
print("Annotations generated", flush=True)

# Extract facial expressions frames
# Due to dependencies incompatibilities, this step is done in a separate environment
run_in_env(f"app/frame_extraction/run_frame_extraction.py {VIDEO_PATH} {FRAMES_PATH} {ANNOTATIONS_PATH}",
           "python_environments/object_detectors_env")
print("Extracted facial expressions frames", flush=True)

facial_expressions_frames_path = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID)

# Generate the base frames embeddings
embeddings_processing.generate_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH, embedder)
print("Frame embeddings generated", flush=True)

# Generate the average and the best frames embeddings
embeddings_processing.generate_average_and_best_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH, embedder)

# Load the base, average, and best frame embeddings
with open(os.path.join(EMBEDDINGS_PATH, "frame_embeddings.json.embeddings"), "rb") as f:
    base_frame_embeddings = CPU_Unpickler(f).load()

with open(os.path.join(EMBEDDINGS_PATH, "average_frame_embeddings.json.embeddings"),
          "rb") as f:
    average_frame_embeddings = CPU_Unpickler(f).load()

with open(os.path.join(EMBEDDINGS_PATH, "best_frame_embeddings.json.embeddings"),
          "rb") as f:
    best_frame_embeddings = CPU_Unpickler(f).load()

print("ENTERING INDEXING LOOP")
for video_id in os.listdir(facial_expressions_frames_path):

    # Read annotations
    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)

        if FACIAL_EXPRESSIONS_ID not in video_annotations:
            continue

        if video_id not in os.listdir(facial_expressions_frames_path):
            continue

        annotations = video_annotations[FACIAL_EXPRESSIONS_ID]["annotations"]

        for annotation in annotations:
            annotation_id = annotation["annotation_id"]
            annotation_value = annotation["value"]
            start_time = annotation["start_time"]
            end_time = annotation["end_time"]
            phrase = annotation["phrase"]

            # Generate a document for indexing in OpenSearch
            doc = gen_doc(
                video_id=video_id,
                annotation_id=annotation_id,
                annotation_value=annotation_value,
                base_frame_embedding=base_frame_embeddings[video_id][annotation_id].tolist(),
                average_frame_embedding=average_frame_embeddings[video_id][annotation_id].tolist(),
                best_frame_embedding=best_frame_embeddings[video_id][annotation_id].tolist(),
                start_time=start_time,
                end_time=end_time,
                phrase=phrase
            )

            # Index the document in OpenSearch
            # opensearch.index_if_not_exists(doc)
            opensearch.delete_doc_and_index(doc)

print("Finished processing videos", flush=True)
