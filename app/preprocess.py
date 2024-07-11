import json
import os
import subprocess
from .eaf_parser import eaf_parser
from .embeddings import embeddings_processing
from .opensearch.opensearch import LGPOpenSearch, gen_doc
from .utils import CPU_Unpickler, RESULTS_PATH, EAF_PATH, VIDEO_PATH, FRAMES_PATH, ANNOTATIONS_PATH, EMBEDDINGS_PATH, \
    FACIAL_EXPRESSIONS_FRAMES_DIR, FACIAL_EXPRESSIONS_ID

# Initialize the OpenSearch client
opensearch = LGPOpenSearch()


def run_in_env(script_path, env_path):
    """ Run a script in a virtual environment """
    activate_script = f'source {env_path}/bin/activate'
    command = f"{activate_script}; python {script_path}; deactivate"
    process = subprocess.Popen(command, shell=True, executable="/bin/bash", stderr=subprocess.PIPE)
    _, err = process.communicate()
    if process.returncode != 0:
        print(f"Error occurred: {err.decode()}")
    process.wait()


""" Preprocess videos, extract frames, generate embeddings and create indexes in OpenSearch """

# Create the videofiles directory if it doesn't exist
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# Generate json file for videos with annotations and timestamps
eaf_parser.parse_eaf_files(EAF_PATH)
print("Annotations generated", flush=True)

# Extract facial expressions frames

# Crate the directory for the frames
if not os.path.exists(FRAMES_PATH):
    os.makedirs(FRAMES_PATH)

# Due to dependencies incompatibilities, this step is done in a separate environment
run_in_env(f"app/frame_extraction/run_frame_extraction.py {VIDEO_PATH} {FRAMES_PATH} {ANNOTATIONS_PATH}",
           "python_environments/object_detectors_env")
print("Extracted facial expressions frames", flush=True)

# Generate embeddings
embeddings_processing.generate_video_embeddings()

# Load the base, average, and best frame embeddings
with open(os.path.join(EMBEDDINGS_PATH, "frame_embeddings.json.embeddings"), "rb") as f:
    base_frame_embeddings = CPU_Unpickler(f).load()

with open(os.path.join(EMBEDDINGS_PATH, "average_frame_embeddings.json.embeddings"), "rb") as f:
    average_frame_embeddings = CPU_Unpickler(f).load()

with open(os.path.join(EMBEDDINGS_PATH, "best_frame_embeddings.json.embeddings"), "rb") as f:
    best_frame_embeddings = CPU_Unpickler(f).load()

with open(os.path.join(EMBEDDINGS_PATH, "summed_frame_embeddings.json.embeddings"), "rb") as f:
    summed_frame_embeddings = CPU_Unpickler(f).load()

with open(os.path.join(EMBEDDINGS_PATH, "annotations_embeddings.json.embeddings"), "rb") as f:
    annotations_embeddings = CPU_Unpickler(f).load()

print("ENTERING INDEXING LOOP", flush=True)
# opensearch.delete_index()
# opensearch.create_index()
for video_id in os.listdir(FACIAL_EXPRESSIONS_FRAMES_DIR):

    # Read annotations
    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)

        if FACIAL_EXPRESSIONS_ID not in video_annotations:
            continue

        if video_id not in os.listdir(FACIAL_EXPRESSIONS_FRAMES_DIR):
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
                base_frame_embedding=base_frame_embeddings[video_id][annotation_id].tolist(),
                average_frame_embedding=average_frame_embeddings[video_id][annotation_id].tolist(),
                best_frame_embedding=best_frame_embeddings[video_id][annotation_id].tolist(),
                summed_frame_embeddings=summed_frame_embeddings[video_id][annotation_id].tolist(),
                annotation_embedding=annotations_embeddings[video_id][annotation_id].tolist(),
            )

            # Index the document in OpenSearch
            # opensearch.index_if_not_exists(doc)
            opensearch.update_doc_and_index(doc)

print("Finished processing videos", flush=True)
