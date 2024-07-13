import argparse
from frame_extractor import FrameExtractor

# Create the parser
parser = argparse.ArgumentParser(description='Process video frames.')

# Call the function with the arguments
extractor = FrameExtractor()
extractor.extract_frames()
