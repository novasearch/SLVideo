import math

from PIL import Image
from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection, YolosImageProcessor, YolosForObjectDetection
import torch
import os


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
        print("CUDA available: ", torch.cuda.is_available())
        print("Number of GPUs: ", torch.cuda.device_count())
        print("GPU: ", torch.cuda.get_device_name(0))

        # Initialize the DETR model for object detection and the Image Processor
        self.crop_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.crop_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.crop_model = self.crop_model.to(self.device)

        print("Person Detector's Device Second Check: ", self.device, flush=True)

        # Initialize the RMBG-1.4 model for image segmentation
        self.segmentation_model = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True,
                                           device=self.device)

    def detect_person(self, images_paths):
        """ Detect a person in an image, crop the person box, detect the person in the
            cropped image using segmentation and save the cropped image """

        # Extract the required parts from the image_paths for the print statements
        base_name = os.path.basename(os.path.dirname(images_paths[0]))
        parent_name = os.path.basename(os.path.dirname(os.path.dirname(images_paths[0])))

        print(f"{parent_name} - {base_name} || Cropping Started", flush=True)

        import time
        start = time.time()

        # If the images length is larger than 64, split the images into same sizes batches smaller than 64
        if len(images_paths) > 64:
            n_batches = math.ceil(len(images_paths) / 64)
            batch_size = math.ceil(len(images_paths) / n_batches)
            images_paths = [images_paths[i:i + batch_size] for i in range(0, len(images_paths), batch_size)]

            for images_paths in images_paths:
                cropped_images = self.detect_person_box(images_paths)
                masked_images = self.detect_person_segmentation(cropped_images)

                for image, image_path in zip(masked_images, images_paths):
                    image.save(image_path)
                    image.close()
        else:
            cropped_images = self.detect_person_box(images_paths)
            masked_images = self.detect_person_segmentation(cropped_images)

            for image, image_path in zip(masked_images, images_paths):
                image.save(image_path)
                image.close()


        end = time.time()
        print(f"{parent_name} - {base_name} || Cropping Finished in {end - start} seconds", flush=True)

    def detect_person_box(self, images_paths):
        """ Detect a person in an image, crop the person box and save the cropped image """

        # Load the images
        images = [Image.open(image_path) for image_path in images_paths]

        inputs = self.crop_processor(images=images, return_tensors="pt")

        # Move the inputs to the GPU
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        outputs = self.crop_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1] for image in images])
        results = self.crop_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)

        cropped_images = []

        for result, image in zip(results, images):
            person_boxes = [box for box, label in zip(result["boxes"], result["labels"]) if label == 1]

            for box in person_boxes:
                box = [round(i.item()) for i in box]
                cropped_image = image.crop(box)
                cropped_images.append(cropped_image)

            image.close()

        return cropped_images

    def detect_person_segmentation(self, cropped_images):
        """ Detect the person in an image using segmentation, crop the person mask and save the cropped image """

        # pillow_mask = self.segmentation_model(image_paths, return_mask=True)  # outputs a pillow mask
        pillow_images = self.segmentation_model(cropped_images)  # applies mask on input and returns a pillow image

        return pillow_images
