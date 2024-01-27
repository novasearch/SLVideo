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


def gen_doc(frame_id: str, video_id: str, annotation_id: str, path: str, timestamp, linguistic_type_ref: str,
            annotation: str, frame_embedding, annotation_embedding):
    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "annotation_id": annotation_id,
        "path": path,
        "timestamp": timestamp,
        "linguistic_type_ref": linguistic_type_ref,
        "annotation": annotation,
        "frame_embedding": frame_embedding,
        "annotation_embedding": annotation_embedding
    }


def preprocess_videos():
    """ Preprocess videos (extract the facial expressions frames)
    and generate embeddings. All the results are saved in the static/videofiles folder. """

    # Generate json file for videos with annotations and timestamps
    eaf_parser.parse_eaf_files(EAF_PATH)
    print("Annotations generated")
    time.sleep(1)

    # Extract facial expressions frames
    frame_extraction.extract_frames(VIDEO_PATH, FRAMES_PATH, ANNOTATIONS_PATH)
    print("Extracted facial expressions frames")
    time.sleep(1)

    facial_expressions_frames_path = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID)

    # Generate the base frames embeddings
    generate_embeddings.generate_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH)
    print("Frame embeddings generated")
    time.sleep(1)

    # Generate the average frames embeddings
    generate_embeddings.generate_average_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH)
    print("Average frame embeddings generated")
    time.sleep(1)

    # Generate the best frame embeddings
    generate_embeddings.generate_best_frame_embeddings(facial_expressions_frames_path, EMBEDDINGS_PATH)
    print("Best frame embeddings generated")
    time.sleep(1)

    print("Reading base frames embeddings")
    with open(os.path.join(EMBEDDINGS_PATH, "frame_embeddings.json.embeddings"), "rb") as f:
        base_frame_embeddings = CPU_Unpickler(f).load()

    print("Reading average frame embeddings")
    with open(os.path.join(EMBEDDINGS_PATH, "average_frame_embeddings.json.embeddings"), "rb") as f:
        average_frame_embeddings = CPU_Unpickler(f).load()

    print("Reading best frame embeddings")
    with open(os.path.join(EMBEDDINGS_PATH, "best_frame_embeddings.json.embeddings"), "rb") as f:
        best_frame_embeddings = CPU_Unpickler(f).load()


"""
# If index doesn't exist, create it
opensearch.create_index()

# Generate annotation embeddings
generate_annotation_embeddings(ANNOTATIONS_PATH, RESULTS_PATH)
print("Annotation embeddings generated")
time.sleep(1)

# Generate frame embeddings
generate_frame_embeddings(FRAMES_PATH, RESULTS_PATH)
print("Frame embeddings generated")
time.sleep(1)

print("Reading annotations embeddings")
with open(os.path.join(RESULTS_PATH, "annotation_embeddings.json.embeddings"), "rb") as f:
    annotation_embeddings = CPU_Unpickler(f).load()

print("Reading image embeddings")
with open(os.path.join(RESULTS_PATH, "frame_embeddings.json.embeddings"), "rb") as f:
    frame_embeddings = CPU_Unpickler(f).load()

print("ENTERING INDEXING LOOP")
# Read frames
for video_id in os.listdir(FRAMES_PATH):
    video_path = os.path.join(FRAMES_PATH, video_id)

    # Read annotations
    with open(os.path.join(ANNOTATIONS_PATH, f"{video_id}.json"), "r") as f:
        video_annotations = json.load(f)

        for frame in os.listdir(video_path):

            with open(os.path.join(RESULTS_PATH,"timestamps", f"{video_id}.json", ), "r") as f:
                timestamps = json.load(f)

                frame_id = os.path.splitext(frame)[0]

                # Find the annotation corresponding to this frame
                # TODO: Check if it is needed to iterate over all annotations tiers
                annotations = video_annotations["GLOSA_P1_EXPRESSAO"]["annotations"]

                # TODO: Check how are the timestamps created
                for annotation in annotations:
                    if annotation["start_time"] <= timestamps[frame_id] <= annotation["end_time"]:
                        annotation_id = annotation["annotation_id"]
                        linguistic_type_ref = video_annotations["GLOSA_P1_EXPRESSAO"]["tier_id"]
                        annotation = annotation["value"]

                        doc = gen_doc(
                            frame_id=frame_id,
                            video_id=video_id,
                            annotation_id=annotation_id,
                            path=os.path.join(video_path, frame),
                            timestamp=timestamps[frame_id],
                            linguistic_type_ref=linguistic_type_ref,
                            annotation=annotation,
                            frame_embedding=frame_embeddings[video_id][frame_id].tolist(),
                            annotation_embedding=annotation_embeddings[video_id]["GLOSA_P1_EXPRESSAO"][annotation_id].tolist()
                            # TODO
                        )

                        opensearch.index_if_not_exists(doc)

                        print("###########")
                        print(doc)
                        print("###########")
                        print("\n\n")

                        break
"""
