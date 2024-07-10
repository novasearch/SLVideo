# Define the paths for the results, videos, eaf files, annotations, frames, and embeddings
import os

RESULTS_PATH = "app/static/videofiles"  # sys.argv[exp9]
VIDEO_PATH = "app/static/videofiles/mp4"  # sys.argv[exp10]
EAF_PATH = "app/static/videofiles/eaf"  # sys.argv[exp11]
ANNOTATIONS_PATH = "app/static/videofiles/annotations"  # sys.argv[4]
FRAMES_PATH = "app/static/videofiles/frames"  # sys.argv[5]
EMBEDDINGS_PATH = "app/embeddings"  # sys.argv[6]

PHRASES_ID = "LP_P1 transcrição livre"
FACIAL_EXPRESSIONS_ID = "GLOSA_P1_EXPRESSAO"

FACIAL_EXPRESSIONS_FRAMES_DIR = os.path.join(FRAMES_PATH, FACIAL_EXPRESSIONS_ID)
