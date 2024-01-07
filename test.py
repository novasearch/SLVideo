import sys
from subprocess import call

from eaf_parser.eaf_parser import EAFParser
from embeddings.generate_embeddings import generate_frame_embeddings, generate_annotation_embeddings

RESULTS_PATH = "test_embeddings" #sys.argv[1]
FRAMES_PATH = "frame_extraction/results"  #sys.argv[2]

#parser = EAFParser()

#parser.create_data_dict()

#parser.save_data_dict_json()

#generate_annotation_embeddings("test.json", "test_embeddings")

rc = call(["frame_extraction/parse_video_frames.sh"], cwd=".")
if rc != 0:
    print("Error: frame extraction failed")
    sys.exit(1)

generate_frame_embeddings(FRAMES_PATH, RESULTS_PATH)