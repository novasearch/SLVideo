import sys
import os
import object_detector

# Call the function with the arguments
image_paths = sys.argv[1]

od = object_detector.ObjectDetector()
od.detect_person(image_paths.split(","))