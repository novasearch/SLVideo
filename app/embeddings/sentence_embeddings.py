from sentence_transformers import SentenceTransformer
import open_clip

from PIL import Image
import torch


class Embedder():

    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print("Embedder's Device: ", self.device, flush=True)

        self.model = SentenceTransformer('clip-ViT-B-32', device=self.device)

        #self.model, preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:hiaac-nlp/CAPIVARA')
        #self.tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')

    # Encode text
    def text_encode(self, texts):
        """ clip-ViT-B-32 """
        model_output = self.model.encode(texts, convert_to_tensor=True)

        """ CAPIVARA """
        #texts = self.tokenizer(texts)
        #model_output = self.model.encode_text(texts)

        # Flatten the tensor if it's more than 1D
        if model_output.dim() > 1:
            model_output = model_output.view(-1)

        return model_output

    # Encode image (frame extraction results)
    def image_encode(self, path):
        image = Image.open(path)

        """ clip-ViT-B-32 """
        model_output = self.model.encode(image, convert_to_tensor=True)

        """ CAPIVARA """
        #image = Image.open(path)
        #image = self.preprocess_val(image).unsqueeze(0)
        #model_output = self.model.encode_image(image)

        # Flatten the tensor if it's more than 1D
        if model_output.dim() > 1:
            model_output = model_output.view(-1)

        return model_output
