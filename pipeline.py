import json
import os
import pickle
import torch
import io
import sys
import time
from subprocess import call

from eaf_parser.eaf_parser import EAFParser
from embeddings.generate_embeddings import generate_frame_embeddings, generate_annotation_embeddings
from opensearch.opensearch import LGPOpenSearch

RESULTS_PATH = "frame_extraction/results"  # sys.argv[1]
VIDEO_PATH = "videofiles"  # sys.argv[2]

opensearch = LGPOpenSearch()
parser = EAFParser()


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


for video in os.listdir(VIDEO_PATH):
    frame_path = os.path.join(VIDEO_PATH, video)

    # Extracts results from video
    rc = call(["frame_extraction/parse_video_frames.sh", os.path.join(VIDEO_PATH, video + ".mp4"), RESULTS_PATH],
              cwd=".")
    if rc != 0:
        print("Error: frame extraction failed")
        sys.exit(1)
    print("Frames extracted from video" + video)
    time.sleep(1)

    # Generate json file for video with annotations and timestamps
    video_annotations = parser.create_data_dict(os.path.join(VIDEO_PATH, video + ".eaf"))
    print("Annotations generated for video" + video)
    time.sleep(1)

    # Generate annotation embeddings
    generate_annotation_embeddings(video_annotations, os.path.join(RESULTS_PATH, video + "_embeddings"))
    print("Annotation embeddings generated for video" + video)
    time.sleep(1)

    # Generate frame embeddings
    generate_frame_embeddings(frame_path, os.path.join(RESULTS_PATH, video + "_embeddings", "results"))
    print("Frame embeddings generated for video" + video)
    time.sleep(1)

    print("Reading embeddings for video" + video)
    with open("./frame_caption_embeddings.json.embeddings", "rb") as f:
        annotation_embeddings = CPU_Unpickler(f).load()

    print("Reading image embeddings for video" + video)
    with open("./frame_image_embeddings.json.embeddings", "rb") as f:
        frame_embeddings = CPU_Unpickler(f).load()

    for frame in os.listdir(frame_path):
        # Read timestamps
        timestamps = {}

        with open(os.path.join("timestamps", f"{video}.json", ), "r") as f:
            timestamps = json.load(f)

        frame_id = os.path.splitext(frame)[0]

        # Find annotation for this frame
        # TODO: Check if it is needed to iterate over all annotations tiers
        annotations = video_annotations["LP_P1 transcrição livre"]["annotations"]

        # TODO: Check how are the timestamps created
        for annotation in annotations:
            if annotation["start_time"] <= timestamps[frame_id] <= annotation["end_time"]:
                annotation_id = annotation["annotation_id"]
                linguistic_type_ref = video_annotations["LP_P1 transcrição livre"]["tier_id"]
                annotation = annotation["value"]
                break

        doc = gen_doc(
            frame_id=frame_id,
            video_id=video,
            annotation_id=annotation_id,
            path=os.path.join(frame_path, frame),
            timestamp=timestamps[frame_id],
            linguistic_type_ref=linguistic_type_ref,
            annotation=annotation,
            frame_embedding=frame_embeddings[frame_id].tolist(),
            annotation_embedding=annotation_embeddings["LP_P1 transcrição livre"][annotation_id].tolist()  # TODO
        )

        opensearch.index_if_not_exists(doc)

        print("###########")
        print(doc)
        print("###########")
        print("\n\n")
