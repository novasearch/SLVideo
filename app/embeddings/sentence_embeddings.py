from sentence_transformers import SentenceTransformer
import open_clip

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
        #self.model = SentenceTransformer('clip-ViT-B-32', device=device)

        self.model, preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:hiaac-nlp/CAPIVARA')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')

    # Encode text
    def text_encode(self, texts):
        """ clip-ViT-B-32
        model_output = self.model.encode(texts, convert_to_tensor=True)
        """

        """ CAPIVARA """
        texts = self.tokenizer(texts, return_tensors="pt", padding=True)
        model_output = self.model.encode_text(texts)

        return model_output

    # Encode image (frame extraction results)
    def image_encode(self, path):
        """ clip-ViT-B-32
        model_output = self.model.encode(Image.open(path),convert_to_tensor=True)
        """

        """ CAPIVARA """
        image = Image.open(path)
        image = self.preprocess_val(image).unsqueeze(0)
        model_output = self.model.encode_image(image)

        return model_output
