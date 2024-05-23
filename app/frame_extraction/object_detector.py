import os

from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image


class ObjectDetector:

    def __init__(self):
        # Check if a GPU is available and if so, move the model to the GPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print("Person Detector's Device: ", self.device, flush=True)

        # Initialize the DETR model for object detection and the Image Processor
        self.crop_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.crop_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.crop_model = self.crop_model.to(self.device)

        # Initialize the RMBG-1.4 model for image segmentation
        self.segmentation_model = None #pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True,
                                       #    device=self.device)

    def detect_person(self, image_path):
        """ Detect a person in an image, crop the person box, detect the person in the
            cropped image using segmentation and save the cropped image """

        cropped_image = self.detect_person_box(image_path)
        if cropped_image is None:
            return
        masked_image = self.detect_person_segmentation(cropped_image)

        masked_image.save(image_path)
        masked_image.close()

    def detect_person_box(self, image_path):
        """ Detect a person in an image, crop the person box and save the cropped image """
        image = Image.open(image_path)

        inputs = self.crop_processor(images=image, return_tensors="pt")

        # Move the inputs to the GPU
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        outputs = self.crop_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.crop_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        person_objects = [result for result in zip(results["scores"], results["labels"], results["boxes"]) if
                          result[1] == 1]

        cropped_image = None

        for person in person_objects:
            box = person[2]
            box = [round(i.item()) for i in box]
            cropped_image = image.crop(box)

        image.close()

        return cropped_image

    def detect_person_segmentation(self, cropped_image):
        """ Detect the person in an image using segmentation, crop the person mask and save the cropped image """

        pillow_image = self.segmentation_model(cropped_image)  # applies mask on input and returns a pillow image

        return pillow_image
