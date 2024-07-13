import argparse
from frames_processing import FrameExtractor

# Create the parser
parser = argparse.ArgumentParser(description='Process video frames.')

# Call the function with the arguments
extractor = FrameExtractor()
extractor.extract_frames()
