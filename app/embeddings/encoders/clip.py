from PIL import Image

from app.embeddings.encoders.abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer


class ClipEncoder(AbstractEncoder):
    """ Encoder class to encode text and image using the clip-Vit-B-32 model """

    def __init__(self, device):
        self.model = SentenceTransformer('clip-ViT-B-32', device=device)
        self.model = self.model.to(device)

    def text_encode(self, text):
        model_output = self.model.encode(text, convert_to_tensor=True)
        if model_output.dim() > 1:
            model_output = model_output.view(-1)
        return model_output

    def image_encode(self, path):
        image = Image.open(path)
        model_output = self.model.encode(image, convert_to_tensor=True)
        if model_output.dim() > 1:
            model_output = model_output.view(-1)
        return model_output
