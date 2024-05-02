from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
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

        # Initialize the DETR model for object detection and the Image Processor
        self.crop_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.crop_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.crop_model = self.crop_model.to(self.device)

        # Initialize the RMBG-1.4 model for image segmentation
        self.segmentation_model = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=self.device)

    def detect_person(self, image_paths):
        """ Detect a person in an image, crop the person box, detect the person in the
            cropped image using segmentation and save the cropped image """

        base_name = os.path.basename(os.path.dirname(image_paths[0]))
        parent_name = os.path.basename(os.path.dirname(os.path.dirname(image_paths[0])))

        print(f"{parent_name} - {base_name} || Cropping Started", flush=True)
        self.detect_person_box(image_paths)
        self.detect_person_segmentation(image_paths)
        print(f"{parent_name} - {base_name} || Cropping Finished", flush=True)

    def detect_person_box(self, image_paths):
        """ Detect a person in an image, crop the person box and save the cropped image """

        # load images
        images = [Image.open(image_path) for image_path in image_paths]

        inputs = self.crop_processor(images=images, return_tensors="pt")

        # Move the inputs to the GPU
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        outputs = self.crop_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1] for image in images])
        results = self.crop_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)

        for result, image_path in zip(results, image_paths):
            person_boxes = [box for box, label in zip(result["boxes"], result["labels"]) if label == 1]

            for box in person_boxes:
                box = [round(i.item()) for i in box]
                image = Image.open(image_path)
                cropped_image = image.crop(box)
                cropped_image.save(image_path)

    def detect_person_segmentation(self, image_paths):
        """ Detect the person in an image using segmentation, crop the person mask and save the cropped image """
        for image_path in image_paths:
            img = Image.open(image_path)

            # Convert RGBA images to RGB
            if img.mode == 'RGBA':
                rgb_image = img.convert('RGB')
                rgb_image.save(image_path)

        # pillow_mask = self.segmentation_model(image_paths, return_mask=True)  # outputs a pillow mask
        pillow_images = self.segmentation_model(image_paths)  # applies mask on input and returns a pillow image

        for pillow_image, image_path in zip(pillow_images, image_paths):
            pillow_image.save(image_path)
