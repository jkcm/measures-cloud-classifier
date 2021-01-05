# -*- coding: utf-8 -*-
import pandas as pd
import ast
import numpy as np
import datetime as dt

test_classification_file = r'/home/disk/eos4/jkcm/Data/MEASURES/zooniverse/classification_datasets/classify_128km-classifications_2019-03-28.csv'


def subject_data_parser(str):
    str = str.replace('null', 'None')
    parsed_dict =  ast.literal_eval(str)
    assert len(parsed_dict.keys()) == 1
    return list(parsed_dict.values())[0]


def annotations_parser_v3271(annotation_string):
    annotation_string = annotation_string.replace('"value":null}', '"value":"Other cloud type, not in previous list."}')
    label_map = {
        'Solid Stratus (no cells)': 'solid_stratus',
        'Closed-cellular MCC': 'closed_mcc',
        'Open-cellular MCC': 'open_mcc',
        'Disorganized MCC': 'disorg_mcc',
        'Suppressed cumulus': 'suppressed_cu',
        'Clustered cumulus': 'clustered_cu',
        'Mixed/No dominant type': 'bad_no_dominant_type',
        'Too much sun glint': 'bad_sun_glint',
        'Too much \'bowtie\' (striping from high viewing angle)': 'bad_bowtie',
        'Other cloud type, not in previous list.': 'bad_other',
        'Other cloud type, not in previous list,': 'bad_other',
        'Polar (sea ice/night time)': 'bad_polar',
        'Scattered convection': 'suppressed_cu',
        'Mixed': 'bad_other',
        'Other': 'bad_other',
        'Lined convection': 'bad_other'}
    try:
        parsed_dict = {x['task']: x['value'] for x in ast.literal_eval(annotation_string)}
    except ValueError:
        return 'bad_other'
    if parsed_dict['T0'] == "Other/no dominant type/can't tell":
        if not parsed_dict['T2']:
            print('null value!')
            return label_map['Other cloud type, not in previous list.']
        return label_map[parsed_dict['T2']]
    else:
        if parsed_dict['T0'] not in label_map.keys():
            return 'bad_other'
        return label_map[parsed_dict['T0']]

def tidy_parser(annotation_string, basic_parser=annotations_parser_v3271):
    rough_string = basic_parser(annotation_string)
    nice_map = {'solid_stratus': 'Stratus',
        'closed_mcc': 'Closed-cell MCC',
        'open_mcc': 'Open-cell MCC',
        'disorg_mcc': 'Disorg. MCC',
        'suppressed_cu': 'Suppressed Cu',
        'clustered_cu': 'Clustered Cu',
        'bad_no_dominant_type': 'No Dominant Type',
        'bad_sun_glint': 'Other Issue',
        'bad_bowtie': 'Other Issue',
        'bad_other': 'Other Issue',
        'bad_polar': 'Other Issue',
        'Scattered convection': 'Suppressed Cu'}
    return nice_map[rough_string]
    
    
annotations_parser = annotations_parser_v3271

def count_number_of_classifications(data):
    class_count = {}#{'total': len(data)}
    for label in set(data['label'].values):
        class_count[label] = sum(data['label'].values == label)
    return class_count

def read_and_parse_classifications(csv_file, annotations_parser=annotations_parser, parse_subject_data=False):
    columns_to_ignore = ['gold_standard', 'expert', 'metadata', 'user_ip']
    data = pd.read_csv(csv_file).drop(columns=columns_to_ignore)
    data = data.drop(index=np.nonzero(data['workflow_version'].values<32.71)[0])
    data['label'] = [annotations_parser(i) for i in data['annotations'].values]
    if parse_subject_data:
        data['subject_data'] = [subject_data_parser(i) for i in data['subject_data'].values]
    return data