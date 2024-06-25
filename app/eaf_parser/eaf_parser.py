import json
import os
import webvtt
import xml.etree.ElementTree as ET

ANNOTATIONS_PATH = "app/static/videofiles/annotations"
CAPTIONS_PATH = "app/static/videofiles/captions"


def parse_eaf_files(eaf_dir):
    """ Parses the eaf files and extracts the data from it """

    for eaf in os.listdir(eaf_dir):
        videoname, extension = os.path.splitext(eaf)
        video_annotations_path = os.path.join(ANNOTATIONS_PATH, videoname + '.json')

        if os.path.isfile(video_annotations_path) or extension != '.eaf':
            continue

        data_dict = {}

        with open(os.path.join(eaf_dir, eaf), "r", encoding='utf-8') as file:
            file_content = file.read()

            root = ET.fromstring(file_content)

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


def convert_milliseconds_to_time_format(milliseconds):
    """ Converts milliseconds to the WebVTT time format (HH:MM:SS.mmm) """
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
