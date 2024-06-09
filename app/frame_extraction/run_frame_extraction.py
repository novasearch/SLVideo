import argparse
from frame_extractor import FrameExtractor

# Create the parser
parser = argparse.ArgumentParser(description='Process video frames.')

# Add the arguments
parser.add_argument('VIDEO_PATH', type=str, help='The path to the video files')
parser.add_argument('FRAMES_PATH', type=str, help='The path to save the frames')
parser.add_argument('ANNOTATIONS_PATH', type=str, help='The path to the annotations')

# Parse the arguments
args = parser.parse_args()

# Call the function with the arguments
extractor = FrameExtractor()
extractor.extract_frames(args.VIDEO_PATH, args.FRAMES_PATH, args.ANNOTATIONS_PATH)
