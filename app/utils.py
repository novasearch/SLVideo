# Description: This file contains the global variables, constants and functions used throughout the application.
import io
import pickle
import torch
import os
from app.embeddings import embeddings_processing
from app.opensearch.opensearch import LGPOpenSearch

# Constants
RESULTS_PATH = "app/static/videofiles"  # sys.argv[exp9]
VIDEO_PATH = "app/static/videofiles/mp4"  # sys.argv[exp10]
EAF_PATH = "app/static/videofiles/eaf"  # sys.argv[exp11]
ANNOTATIONS_PATH = "app/static/videofiles/annotations"  # sys.argv[4]
FRAMES_PATH = "app/static/videofiles/frames"  # sys.argv[5]
EMBEDDINGS_PATH = "app/embeddings"  # sys.argv[6]
CAPTIONS_PATH = "app/static/videofiles/captions"

PHRASES_ID = "LP_P1 transcrição livre"
FACIAL_EXPRESSIONS_ID = "GLOSA_P1_EXPRESSAO"
FACIAL_EXPRESSIONS_ID_2 = "GLOSA_P1_EXPRESSÃO"

PHRASES_FRAMES_DIR = os.path.join(FRAMES_PATH, PHRASES_ID)
FACIAL_EXPRESSIONS_FRAMES_DIR = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID)

BASE_FRAMES_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_PATH, 'frame_embeddings.json.embeddings')
AVERAGE_FRAMES_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_PATH, 'average_frame_embeddings.json.embeddings')
BEST_FRAMES_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_PATH, 'best_frame_embeddings.json.embeddings')
SUMMED_FRAMES_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_PATH, 'summed_frame_embeddings.json.embeddings')
ALL_FRAMES_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_PATH, 'all_frame_embeddings.json.embeddings')
ANNOTATIONS_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_PATH, 'annotations_embeddings.json.embeddings')

# Embedder to be used throughout the application
embedder = embeddings_processing.Embedder(check_gpu=False)

# OpenSearch instance to be used throughout the application
opensearch = LGPOpenSearch()


# Custom Unpickler to load torch tensors on CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
