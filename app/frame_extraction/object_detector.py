from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image


class ObjectDetector:

    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def detect_person(self, image_path):
        print("detect_person function started")
        """ Detect a person in an image, crop the person and save the cropped image """
        image = Image.open(image_path)
        print(f"Image opened: {image_path}")

        inputs = self.processor(images=image, return_tensors="pt")
        print("Inputs processed")
        outputs = self.model(**inputs)
        print("Model outputs obtained")

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        print("Post processing completed")

        person_objects = [result for result in zip(results["scores"], results["labels"], results["boxes"]) if
                          result[1] == 1]
        print(f"Found {len(person_objects)} person objects")

        for person in person_objects:
            box = person[2]
            box = [round(i.item()) for i in box]
            cropped_image = image.crop(box)
            cropped_image.save(image_path)
            print(f"Saved cropped image: {image_path}")

        print("detect_person function completed")