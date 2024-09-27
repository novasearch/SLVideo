import torch

from app.embeddings.encoders.clip import ClipEncoder
from app.embeddings.encoders.capivara import CapivaraEncoder


class Embedder:
    """
    Embedder class to encode text and image and generate its embeddings using one of two CLIP models:
    - clip-ViT-B-32: the image and text model CLIP from OpenAI
    - CAPIVARA: optimized for texts written in Portuguese
    """

    def __init__(self, check_gpu):

        if check_gpu:
            # Check if a GPU is available and if so, move the model to the GPU
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        print("Embedder's Device: ", self.device, flush=True)

        # Select the model to be used
        self.encoder = CapivaraEncoder(self.device)

    def text_encode(self, text):
        """ Encode text and generate its embeddings  using the selected model """
        return self.encoder.text_encode(text)

    def image_encode(self, path):
        """ Encode image and generate its embeddings using the selected model """
        return self.encoder.image_encode(path)
