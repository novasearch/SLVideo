from PIL import Image
from app.embeddings.encoders.abstract_encoder import AbstractEncoder
import open_clip


class CapivaraEncoder(AbstractEncoder):
    """ Encoder class to encode text and image using the CAPIVARA model """

    def __init__(self, device):
        self.device = device
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:hiaac-nlp/CAPIVARA')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')
        self.model = self.model.to(self.device)

    def text_encode(self, text):
        text = self.tokenizer(text)
        text = text.to(self.device)
        model_output = self.model.encode_text(text)
        if model_output.dim() > 1:
            model_output = model_output.view(-1)
        return model_output

    def image_encode(self, path):
        image = Image.open(path)
        image = self.preprocess_val(image).unsqueeze(0)
        image = image.to(self.device)
        model_output = self.model.encode_image(image)
        if model_output.dim() > 1:
            model_output = model_output.view(-1)
        return model_output
