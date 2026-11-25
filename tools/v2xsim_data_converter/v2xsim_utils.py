import json
from tqdm import tqdm
import os
import numpy as np

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)


def process_instance(v2xsim_instances, v2xsim_sample_annotations):
    annotations_mappings = {}
    for annotation in v2xsim_sample_annotations:
        annotations_mappings[annotation['token']] = annotation
    
    for instance in tqdm(v2xsim_instances):
        instance_token = instance['token']
        first_annotation_token = instance['first_annotation_token']
        nbr_annotations = instance['nbr_annotations']

        current_annotation_token = first_annotation_token
        for ii in range(nbr_annotations - 2):
            current_annotation_token = annotations_mappings[current_annotation_token]['next']
        
        current_annotation_token_next = annotations_mappings[current_annotation_token]['next']
        if current_annotation_token_next not in annotations_mappings.keys():
            print("There is wrong instance token: ", instance_token)
            instance['nbr_annotations'] = nbr_annotations - 1
            instance['last_annotation_token'] = current_annotation_token

            annotations_mappings[current_annotation_token]['next'] = ''

    v2xsim_instances_new = v2xsim_instances
    v2xsim_sample_annotations_new = []
    for key in annotations_mappings.keys():
        v2xsim_sample_annotations_new.append(annotations_mappings[key])

    return v2xsim_instances_new, v2xsim_sample_annotations_new


def process_sample_annotations(v2xsim_sample_annotations):
    for annotation in tqdm(v2xsim_sample_annotations):
        annotation['num_lidar_pts'] = 1
        annotation['num_radar_pts'] = 1

        if not all (np.array(annotation['size'])) > 0.0:
            if not annotation['size'][0] > 0.0:
                annotation['size'][0] = 1.0
            else:
                print("There is a except case!", annotation)
                os._exit()

    return v2xsim_sample_annotations


def process_timestamp(v2xsim_sample, v2xsim_sample_data, v2xsim_ego_pose):
    if v2xsim_sample[0]['timestamp'] < 1000:
        sample_timestamp_mappings = {}
        egopose_timestamp_mappings = {}
        for spl_data in tqdm(v2xsim_sample_data):
            scene_num = int(spl_data['filename'].split('/')[-1].split('_')[1])
            new_timestamp = (spl_data['timestamp'] + scene_num * 1000) / 5.0 * 1e6
            spl_data['timestamp'] = new_timestamp

            sample_timestamp_mappings[spl_data['sample_token']] = new_timestamp
            egopose_timestamp_mappings[spl_data['ego_pose_token']] = new_timestamp
        for spl in tqdm(v2xsim_sample):
            spl['timestamp'] = sample_timestamp_mappings[spl['token']]
        for pose in tqdm(v2xsim_ego_pose):
            pose['timestamp'] = egopose_timestamp_mappings[pose['token']]

    return v2xsim_sample, v2xsim_sample_data, v2xsim_ego_pose


def process_log(v2xsim_log):
    for log in v2xsim_log:
        log['location'] = "yizhuang06"

    return v2xsim_log
