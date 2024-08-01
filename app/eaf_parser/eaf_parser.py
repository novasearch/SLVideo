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


def edit_annotation(video_id, tier_id, annotation_id, start_time, end_time, value):
    """ Edit an annotation in the video EAF file """
    video_eaf = os.path.join(EAF_PATH, video_id + '.eaf')

    with open(video_eaf, "r", encoding='utf-8') as file:
        file_content = file.read()

        root = ET.fromstring(file_content)

        # Get the time slots
        time_slots = root.findall('TIME_ORDER/TIME_SLOT')
        time_slot_1 = None
        time_slot_2 = None

        # Edit the annotation
        tier_annotations = root.findall('TIER[@TIER_ID="' + tier_id + '"]/')
        parent_tier = root.find('TIER[@TIER_ID="' + tier_id + '"]').attrib.get('PARENT_REF')
        for outer_tag in tier_annotations:
            annotation = outer_tag.find('*')  # Get the first child element
            if annotation.attrib['ANNOTATION_ID'] == annotation_id:
                annotation.find('ANNOTATION_VALUE').text = value

                # If the annotation is a ref annotation, get the parent annotation's time slots
                if parent_tier is not None:
                    ref_id = annotation.attrib.get('ANNOTATION_REF')
                    ref_annotation = root.findall('TIER[@TIER_ID="' + parent_tier + '"]/')
                    for ref_outer_tag in ref_annotation:
                        ref_annotation = ref_outer_tag.find('*')
                        if ref_annotation.attrib['ANNOTATION_ID'] == ref_id:
                            time_slot_1 = ref_annotation.attrib.get('TIME_SLOT_REF1')
                            time_slot_2 = ref_annotation.attrib.get('TIME_SLOT_REF2')
                else:
                    time_slot_1 = annotation.attrib.get('TIME_SLOT_REF1')
                    time_slot_2 = annotation.attrib.get('TIME_SLOT_REF2')

        # Edit the time slots
        for time_slot in time_slots:
            if time_slot.attrib['TIME_SLOT_ID'] == time_slot_1:
                time_slot.attrib['TIME_VALUE'] = str(start_time)
            elif time_slot.attrib['TIME_SLOT_ID'] == time_slot_2:
                time_slot.attrib['TIME_VALUE'] = str(end_time)

        with open(video_eaf, "w", encoding='utf-8') as updated_file:
            updated_file.write(ET.tostring(root, encoding='unicode'))


def add_annotation(video_id, new_annotation_id, tier_id, new_start_time, new_end_time, value, phrase):
    """ Adds an annotation to the video annotations EAF file """
    video_eaf = os.path.join(EAF_PATH, video_id + '.eaf')
    file = ET.parse(video_eaf)
    root = file.getroot()

    start_time = str(new_start_time)
    end_time = str(new_end_time)

    time_slots = root.findall('TIME_ORDER/TIME_SLOT')

    # Get last time slot id
    last_time_slot_id = int(time_slots[-1].attrib['TIME_SLOT_ID'].split("ts")[1])

    # Create the new time slots
    time_order = root.find('TIME_ORDER')
    time_slot_1 = "ts" + str(last_time_slot_id + 1)
    time_slot_2 = "ts" + str(last_time_slot_id + 2)
    ET.SubElement(time_order, 'TIME_SLOT', {'TIME_SLOT_ID': time_slot_1, 'TIME_VALUE': start_time})
    ET.SubElement(time_order, 'TIME_SLOT', {'TIME_SLOT_ID': time_slot_2, 'TIME_VALUE': end_time})

    # Find the appropriate tier
    tier = root.find(f'TIER[@TIER_ID="{tier_id}"]')
    parent_tier_id = tier.attrib.get('PARENT_REF')

    # Create the new annotation element
    new_outer_tag = ET.SubElement(tier, 'ANNOTATION')

    if parent_tier_id is not None:
        # Add the parent annotation to the new annotation
        parent_tier = root.find(f'TIER[@TIER_ID="{parent_tier_id}"]')
        parent_annotation_id = "a" + str(int(new_annotation_id.split("a")[1]) + 1)
        new_parent_outer_tag = ET.SubElement(parent_tier, 'ANNOTATION')
        new_parent_annotation = ET.SubElement(new_parent_outer_tag, 'ALIGNABLE_ANNOTATION', {
            'ANNOTATION_ID': parent_annotation_id, 'TIME_SLOT_REF1': time_slot_1, 'TIME_SLOT_REF2': time_slot_2
        })
        new_parent_annotation_value = ET.SubElement(new_parent_annotation, 'ANNOTATION_VALUE')
        new_parent_annotation_value.text = value

        new_annotation = ET.SubElement(new_outer_tag, 'REF_ANNOTATION', {
            'ANNOTATION_ID': new_annotation_id, 'ANNOTATION_REF': parent_annotation_id
        })

        # Update the lastUsedAnnotationId in the JSON file
        with open(os.path.join(ANNOTATIONS_PATH, video_id + '.json'), 'r') as f:
            data = json.load(f)
            data['properties']['lastUsedAnnotationId'] = parent_annotation_id.split("a")[1]
        with open(os.path.join(ANNOTATIONS_PATH, video_id + '.json'), 'w') as f:
            json.dump(data, f)
    else:
        new_annotation = ET.SubElement(new_outer_tag, 'ALIGNABLE_ANNOTATION', {
            'ANNOTATION_ID': new_annotation_id, 'TIME_SLOT_REF1': time_slot_1, 'TIME_SLOT_REF2': time_slot_2
        })

    new_annotation_value = ET.SubElement(new_annotation, 'ANNOTATION_VALUE')
    new_annotation_value.text = value

    # Update the lastUsedAnnotationId in the EAF file
    last_used_annotation_id = root.find('HEADER/PROPERTY[@NAME="lastUsedAnnotationId"]')
    if parent_tier_id is not None:
        last_used_annotation_id.text = str(int(new_annotation_id.split("a")[1]) + 1)
    else:
        last_used_annotation_id.text = new_annotation_id.split("a")[1]

    # Write back the updated XML to the EAF file
    with open(video_eaf, "w", encoding='utf-8') as updated_file:
        updated_file.write(ET.tostring(root, encoding='unicode'))


def delete_annotation(video_id, annotation_id, tier_id):
    """ Delete an annotation in the EAF file """
    video_eaf = os.path.join(EAF_PATH, video_id + '.eaf')
    file = ET.parse(video_eaf)
    root = file.getroot()

    # Get the time slots
    time_slots = root.findall('TIME_ORDER/TIME_SLOT')
    time_slot_1 = None
    time_slot_2 = None

    # Delete the annotation
    tier_annotations = root.findall('TIER[@TIER_ID="' + tier_id + '"]/')
    tier = root.find(f'TIER[@TIER_ID="{tier_id}"]')
    for outer_tag in tier_annotations:
        annotation = outer_tag.find('*')
        if annotation.attrib['ANNOTATION_ID'] == annotation_id:
            tier.remove(outer_tag)
            time_slot_1 = annotation.attrib.get('TIME_SLOT_REF1')
            time_slot_2 = annotation.attrib.get('TIME_SLOT_REF2')

    # Delete the time slots if it is an alignable annotation
    if time_slot_1 is not None and time_slot_2 is not None:
        for time_slot in time_slots:
            if time_slot.attrib['TIME_SLOT_ID'] == time_slot_1:
                root.find('TIME_ORDER').remove(time_slot)
            elif time_slot.attrib['TIME_SLOT_ID'] == time_slot_2:
                root.find('TIME_ORDER').remove(time_slot)

    with open(video_eaf, "w", encoding='utf-8') as updated_file:
        updated_file.write(ET.tostring(root, encoding='unicode'))
