from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image


class ObjectDetector:

    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        # Check if a GPU is available and if so, move the model to the GPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = self.model.to(self.device)

    def detect_person(self, image_path):
        """ Detect a person in an image, crop the person and save the cropped image """
        image = Image.open(image_path)

        print("Cropping ", image_path, flush=True)

        inputs = self.processor(images=image, return_tensors="pt")

        # Move the inputs to the GPU
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        person_objects = [result for result in zip(results["scores"], results["labels"], results["boxes"]) if
                          result[1] == 1]

        for person in person_objects:
            box = person[2]
            box = [round(i.item()) for i in box]
            cropped_image = image.crop(box)
            cropped_image.save(image_path)