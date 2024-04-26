from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image


class ObjectDetector:

    def __init__(self):
        """ Initialize the DETR model for object detection and the Image Processor """
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        # Check if a GPU is available and if so, move the model to the GPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print("Person Detector's Device: ", self.device, flush=True)

        self.model = self.model.to(self.device)

    def detect_person(self, image_path):
        """ Detect a person in an image, crop the person box, detect the person in the
            cropped image using segmentation and save the cropped image """

        print("Cropping ", image_path, flush=True)
        self.detect_person_box(image_path)
        self.detect_person_segmentation(image_path)

    def detect_person_box(self, image_path):
        """ Detect a person in an image, crop the person box and save the cropped image """
        image = Image.open(image_path)

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

    def detect_person_segmentation(self, image_path):
        """ Detect the person in an image using segmentation, crop the person mask and save the cropped image """
        img = Image.open(image_path)

        # Convert RGBA images to RGB
        if img.mode == 'RGBA':
            rgb_image = img.convert('RGB')
            rgb_image.save(image_path)

        pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=self.device)
        # pillow_mask = pipe(image_path, return_mask=True)  # outputs a pillow mask
        pillow_image = pipe(image_path)  # applies mask on input and returns a pillow image

        pillow_image.save(image_path)
