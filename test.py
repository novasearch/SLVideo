from eaf_parser.eaf_parser import EAFParser
from embeddings.generate_embeddings import generate_frame_embeddings, generate_annotation_embeddings

parser = EAFParser()

parser.create_data_dict()

parser.save_data_dict_json()

generate_annotation_embeddings("test.json", "test_embeddings")