# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

DISPARITY_MULTIPLIER = 7.0
TIME_BETWEEN_EXAMPLES = 0.05  # sec
EXPERIMENTS = {
    'indoor_flying': [1, 2, 3, 4],
    'outdoor_day': [1, 2],
    'outdoor_night': [1, 2, 3]
}
# Focal length multiplied by baseline [pix * meter].
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
    'outdoor_night': 19.651191,
    'outdoor_day': 19.635287
}
INVALID_DISPARITY = 255
DISPARITY_MAXIMUM = 37
IMAGE_WIDTH = 346
IMAGE_HEIGHT = 260


def create_folders(paths):
    for name, folder in paths.items():
        if isinstance(folder, dict):
            create_folders(folder)
        else:
            if 'folder' in name and not os.path.exists(folder):
                os.makedirs(folder)


def experiment_paths(experiment_name, experiment_number, dataset_root):
    paths = {'cam0': {}, 'cam1': {}}

    paths['experiment_folder'] = os.path.join(
        dataset_root, '%s_%i' % (experiment_name, experiment_number))

    for camera, value in {'cam0': 0, 'cam1': 1}.items():
        paths[camera]['image_folder'] = os.path.join(
            paths['experiment_folder'], 'image%i' % value)
        paths[camera]['image_file'] = os.path.join(
            paths[camera]['image_folder'], '%0.6i.png')
        paths[camera]['event_folder'] = os.path.join(
            paths['experiment_folder'], 'event%i' % value)
        paths[camera]['event_file'] = os.path.join(
            paths[camera]['event_folder'], '%0.6i.npy')

    paths['timestamps_file'] = os.path.join(paths['experiment_folder'],
                                            'timestamps.txt')
    paths['disparity_folder'] = os.path.join(paths['experiment_folder'],
                                             'disparity_image')
    paths['disparity_file'] = os.path.join(paths['disparity_folder'],
                                           '%0.6i.png')
    paths['description'] = os.path.join(dataset_root, 'readme.txt')

    return paths
