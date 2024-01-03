import pickle
import torch
import io

from embeddings.generate_embeddings import generate_frame_embeddings, generate_annotation_embeddings
from opensearch.opensearch import LGPOpenSearch

RESULTS_PATH = ""
FRAMES_PATH = ""

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

# Generate frame embeddings
generate_frame_embeddings(FRAMES_PATH, RESULTS_PATH)