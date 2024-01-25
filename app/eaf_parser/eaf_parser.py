# Description: This file is used to parse the eaf files and extract the data from it.

import xml.etree.ElementTree as ET
import json
import os

ANNOTATIONS_PATH = "app/static/videofiles/annotations"


def parse_eaf_files(eaf_dir):
    for eaf in os.listdir(eaf_dir):
        videoname, extension = os.path.splitext(eaf)
        video_annotations_path = os.path.join(ANNOTATIONS_PATH, videoname + '.json')

        if os.path.isfile(video_annotations_path) or extension != '.eaf':
            continue

        data_dict = {}

        with open(os.path.join(eaf_dir, eaf), "r", encoding='utf-8') as file:
            file_content = file.read()

            # Get the root element
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
                            'end_time': annotation_timestamps[annotation_ref]['end_time']
                        })

            # Check if GLOSA_P1_EXPRESSÃO is in the data_dict and replace for GLOSA_P1_EXPRESSAO
            if 'GLOSA_P1_EXPRESSÃO' in data_dict:
                data_dict['GLOSA_P1_EXPRESSAO'] = data_dict.pop('GLOSA_P1_EXPRESSÃO')


            # Save the data dict as a json file
            with open(video_annotations_path, 'w') as f:
            #with open(os.path.join("../static/videofiles/annotations", videoname + '.json'), 'w', encoding='utf-8') as f:
                json.dump(data_dict, f)

    # Get the data dict as json
    # def get_data_dict_json(self):
    #    return json.dumps(self.data_dict)

    # Saves the data dictionary as a json file
    # def save_data_dict_json(self):
    #    with open('test.json', 'w') as f:
    #        json.dump(self.data_dict, f)


# Execute this script to parse a specific eaf file
#parse_eaf_files("../static/videofiles/eaf")
