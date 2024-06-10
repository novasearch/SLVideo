from sentence_transformers import SentenceTransformer
import open_clip

from PIL import Image
import torch


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

        self.model = SentenceTransformer('clip-ViT-B-32', device=self.device)

        # self.model, _, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:hiaac-nlp/CAPIVARA')
        # self.tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')

        # Move the model to the device
        self.model = self.model.to(self.device)

    def text_encode(self, text):
        """ Encode text and generate its embeddings  using the selected model """

        """ clip-ViT-B-32 """
        model_output = self.model.encode(text, convert_to_tensor=True)

        """ CAPIVARA """
        # text = self.tokenizer(text)
        # model_output = self.model.encode_text(text)

        # Flatten the tensor if it's more than 1D
        if model_output.dim() > 1:
            model_output = model_output.view(-1)

        return model_output

    def image_encode(self, path):
        """ Encode image and generate its embeddings using the selected model """

        image = Image.open(path)

        """ clip-ViT-B-32 """
        model_output = self.model.encode(image, convert_to_tensor=True)

        """ CAPIVARA """
        # image = self.preprocess_val(image).unsqueeze(0)
        # image = image.to(self.device)  # Move the inputs to the GPU
        # model_output = self.model.encode_image(image)

        # Flatten the tensor if it's more than 1D
        if model_output.dim() > 1:
            model_output = model_output.view(-1)

        return model_output
