# Description: This file is used to parse the eaf files and extract the data from it.

import xml.etree.ElementTree as ET
import json


class EAFParser:

    def __init__(self):
        self.video_path = "./videofiles/9.eaf"
        self.data_dict = {}

    def create_data_dict(self):
        with open(self.video_path, "r", encoding='utf-8') as file:
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
            self.data_dict[tier.attrib['TIER_ID']] = {
                'linguistic_type_ref': tier.attrib['LINGUISTIC_TYPE_REF'],
                'tier_id': tier.attrib['TIER_ID'],
                'annotations': [],
                'parent_ref': tier.attrib.get('PARENT_REF')  # Optional
            }

        first_time_slot = int(time_slots_dict['ts1']['time_value'])

        # Get the alignable annotations info
        for tier in self.data_dict:
            annotations = root.findall('TIER[@TIER_ID="' + tier + '"]/ANNOTATION/ALIGNABLE_ANNOTATION')
            for annotation in annotations:
                self.data_dict[tier]['annotations'].append({
                    'annotation_id': annotation.attrib['ANNOTATION_ID'],
                    'start_time': str(int(
                        time_slots_dict[annotation.attrib['TIME_SLOT_REF1']]['time_value']) - first_time_slot),
                    'end_time': str(int(
                        time_slots_dict[annotation.attrib['TIME_SLOT_REF2']]['time_value']) - first_time_slot),
                    'value': annotation.find('ANNOTATION_VALUE').text
                })

        # Get the ref annotations info
        for tier in self.data_dict:
            annotations = root.findall('TIER[@TIER_ID="' + tier + '"]/ANNOTATION/REF_ANNOTATION')
            for annotation in annotations:
                self.data_dict[tier]['annotations'].append({
                    'annotation_id': annotation.attrib['ANNOTATION_ID'],
                    'annotation_ref': annotation.attrib['ANNOTATION_REF'],
                    'value': annotation.find('ANNOTATION_VALUE').text
                })

    # Get the data dict as json
    def get_data_dict_json(self):
        return json.dumps(self.data_dict)

    # Saves the data dictionary as a json file
    def save_data_dict_json(self):
        with open('test.json', 'w') as f:
            json.dump(self.data_dict, f)

    # Get the time slot of the given phrase
    # def get_time_slot_of_phrase(phrase):
    #    for tier in data_dict:
    #        for annotation in data_dict[tier]['annotations']:
    #            if annotation['value'] == phrase:
    #                for time_slot in time_slots_dict:
    #                    if time_slots_dict[time_slot]['time_slot_id'] == annotation['time_slot_ref1']:
    #                        return time_slots_dict[time_slot]['time_value']
    #    return None

    # Get the first time slot value
    # def get_first_time_slot_value():
    #    return time_slots_dict['ts1']['time_value']
