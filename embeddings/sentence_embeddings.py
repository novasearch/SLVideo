from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class Embedder():
    def __init__(self):
        if torch.backends.mps.is_available():
            device = 'mps' 
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.model = SentenceTransformer('clip-ViT-B-32', device=device)

    # Encode text
    def text_encode(self, texts):
        model_output = self.model.encode(texts, convert_to_tensor=True)
        return model_output

    # Encode image (video frames)
    def image_encode(self, path):
        model_output = self.model.encode(Image.open(path),convert_to_tensor=True)
        return model_output