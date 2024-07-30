import json
import os
import subprocess

import webvtt
import xml.etree.ElementTree as ET

from app.utils import ANNOTATIONS_PATH, CAPTIONS_PATH, EAF_PATH, VIDEO_PATH


def parse_eaf_files():
    """ Parses the eaf files and extracts the data from it """

    # Create the annotations and captions directories if they don't exist
    if not os.path.exists(ANNOTATIONS_PATH):
        os.makedirs(ANNOTATIONS_PATH)
    if not os.path.exists(CAPTIONS_PATH):
        os.makedirs(CAPTIONS_PATH)

    for eaf in os.listdir(EAF_PATH):
        videoname, extension = os.path.splitext(eaf)
        video_annotations_path = os.path.join(ANNOTATIONS_PATH, videoname + '.json')

        if os.path.isfile(video_annotations_path) or extension != '.eaf':
            continue

        data_dict = {}

        with open(os.path.join(EAF_PATH, eaf), "r", encoding='utf-8') as file:
            file_content = file.read()

            root = ET.fromstring(file_content)

            # Get the header info
            header_properties = root.findall('HEADER/PROPERTY')
            data_dict['properties'] = {}
            for prop in header_properties:
                data_dict['properties'][prop.attrib['NAME']] = prop.text

            data_dict['properties']['frame_rate'] = get_video_frame_rate(os.path.join(VIDEO_PATH, videoname + '.mp4'))

            # Get the time slots info, cycle through them and save each time slot
            time_slots = root.findall('TIME_ORDER/TIME_SLOT')
            time_slots_dict = {}
            for time_slot in time_slots:
                time_slots_dict[time_slot.attrib['TIME_SLOT_ID']] = {
                    'time_slot_id': time_slot.attrib['TIME_SLOT_ID'],
                    'time_value': time_slot.attrib['TIME_VALUE']
                }

            # Get the tiers info, cycle through them and save each tier
            tiers = root.findall('TIER')
            for tier in tiers:
                data_dict[tier.attrib['TIER_ID']] = {
                    'linguistic_type_ref': tier.attrib['LINGUISTIC_TYPE_REF'],
                    'tier_id': tier.attrib['TIER_ID'],
                    'annotations': [],
                    'parent_ref': tier.attrib.get('PARENT_REF')  # Optional
                }

            # Get the alignable annotations info
            annotation_timestamps = {}
            for tier in data_dict:
                annotations = root.findall('TIER[@TIER_ID="' + tier + '"]/ANNOTATION/ALIGNABLE_ANNOTATION')
                for annotation in annotations:
                    data_dict[tier]['annotations'].append({
                        'annotation_id': annotation.attrib['ANNOTATION_ID'],
                        'start_time': str(int(
                            time_slots_dict[annotation.attrib['TIME_SLOT_REF1']]['time_value'])),
                        'end_time': str(int(
                            time_slots_dict[annotation.attrib['TIME_SLOT_REF2']]['time_value'])),
                        'value': annotation.find('ANNOTATION_VALUE').text
                    })
                    annotation_timestamps[annotation.attrib['ANNOTATION_ID']] = {
                        'start_time': str(int(
                            time_slots_dict[annotation.attrib['TIME_SLOT_REF1']]['time_value'])),
                        'end_time': str(int(
                            time_slots_dict[annotation.attrib['TIME_SLOT_REF2']]['time_value']))
                    }

            # Get the ref annotations info
            for tier in data_dict:
                annotations = root.findall('TIER[@TIER_ID="' + tier + '"]/ANNOTATION/REF_ANNOTATION')
                for annotation in annotations:
                    annotation_ref = annotation.attrib['ANNOTATION_REF']
                    if annotation_ref in annotation_timestamps:
                        data_dict[tier]['annotations'].append({
                            'annotation_id': annotation.attrib['ANNOTATION_ID'],
                            'annotation_ref': annotation_ref,
                            'value': annotation.find('ANNOTATION_VALUE').text,
                            'start_time': annotation_timestamps[annotation_ref]['start_time'],
                            'end_time': annotation_timestamps[annotation_ref]['end_time'],
                            'phrase': 'N/A',
                            'user_rating': 'N/A'
                        })

            # Check if "GLOSA_P1_EXPRESSÃO" is in the data_dict and replace for "GLOSA_P1_EXPRESSAO"
            if 'GLOSA_P1_EXPRESSÃO' in data_dict:
                data_dict['GLOSA_P1_EXPRESSAO'] = data_dict.pop('GLOSA_P1_EXPRESSÃO')

            # Check if ""LP transcrição livre" is in the data_dict and replace for "LP_P1 transcrição livre"
            if 'LP transcrição livre' in data_dict:
                data_dict['LP_P1 transcrição livre'] = data_dict.pop('LP transcrição livre')

            if 'GLOSA_P1_EXPRESSAO' in data_dict:
                # For each GLOSA_P1_EXPRESSAO annotation, add the value of the LP_P1 transcrição livre annotation
                for expression_glosa in data_dict['GLOSA_P1_EXPRESSAO']['annotations']:
                    for phrase in data_dict['LP_P1 transcrição livre']['annotations']:
                        if int(phrase['start_time']) - 10 <= int(expression_glosa['start_time']) and int(
                                phrase['end_time']) + 10 >= int(expression_glosa['end_time']):
                            expression_glosa['phrase'] = phrase['value']
                            break

            # Save the data dict as a json file
            with open(video_annotations_path, 'w') as f:
                json.dump(data_dict, f)

            generate_captions(data_dict['LP_P1 transcrição livre']['annotations'], videoname)


def generate_captions(phrases, videoname):
    """ Generates the captions in WebVTT format from the 'LP_P1 transcrição livre' annotations """
    vtt = webvtt.WebVTT()

    for caption in phrases:
        if caption['value'] == '' or caption['value'] == None:
            continue

        c = webvtt.Caption()

        c.start = convert_milliseconds_to_time_format(int(caption['start_time']))
        c.end = convert_milliseconds_to_time_format(int(caption['end_time']))

        c.text = caption['value']

        vtt.captions.append(c)

    captions_path = os.path.join(CAPTIONS_PATH, videoname + '.vtt')
    vtt.save(captions_path)


def get_video_frame_rate(video_path):
    # Use FFmpeg to extract video information
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
           video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    frame_rate_str = result.stdout.decode('utf-8').strip()
    try:
        # The frame rate is usually in the format of NUM/DENOM (e.g., "30000/1001")
        num, denom = map(int, frame_rate_str.split('/'))
        frame_rate = num / denom
    except ValueError:
        # If the frame rate is a simple number (which is less common)
        frame_rate = float(frame_rate_str)
    return frame_rate


def convert_milliseconds_to_time_format(milliseconds):
    """ Converts milliseconds to the WebVTT time format (HH:MM:SS.mmm) """
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def convert_time_format_to_milliseconds(time_format):
    """ Converts from HH:MM:SS.mmm to milliseconds """
    time_format = time_format.split(':')
    hours = int(time_format[0]) * 3600000
    minutes = int(time_format[1]) * 60000
    seconds = int(time_format[2].split('.')[0]) * 1000
    if '.' in time_format[2]:
        milliseconds = int(time_format[2].split('.')[1])
    else:
        milliseconds = 0
    return hours + minutes + seconds + milliseconds
