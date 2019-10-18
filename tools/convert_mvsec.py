#!/usr/bin/env python2.7
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Downloads MVSEC experiment and converts it to stereo-friendly format.

The original MVSEC is here: https://daniilidis-group.github.io/mvsec/download/.
For every experiment the script produces:

* disparity_image/ - folder with rectified and undistorted disparity images for
           left DAVIS camera, e.g. 000000.png, 0000001.png. The disparities are
           multiplied by 7 and rounded before saving. Unknown disparities
           and disparities > 255 (after the multiplification) are set to 255.
* timestamps.txt - file with synchronization times. Each row corresponds to
                   disparity from "disparity_image/".
* left_event/, right_event/ folder with rectified and undistorted events files,
                     e.g. 000000.npy, 000001.npy. Each event file corresponds
                     to disparity image from "disparity_image/" and contains
                     all events preceding disparity acquisition by 0.05 sec.
                     Each row of the event file contains (timestamp, x, y,
                     polarity). The polarity is encoded as -1 / 1.
                     Events that fall outside of sensor after the
                     rectification are ignored. Spatial coordinates of all
                     events are rounded.
* left_image/, right_image/ folder with rectified and undistorted gray images,
                 e.g. 000000.png, 000001.png. Each image corresponds to
                 disparity image from disparity_image/ and roughly synchronized
                 with the disparity acquisition. These images can be used only
                 for visualization, because synchronization between left and
                 right stereo images is not perfect.

The argument are explained in the main function below.

Example call:

    ./convert_mvsec.py /dataset \
        --experiment "outdoor_night" 1
        --temporal_folder /temp
"""

import click
import tempfile
import pkg_resources
import os
import sys

import cv2
import numpy as np
import cv_bridge

import bag_indexer
import calibration
import downloader

sys.path.append(pkg_resources.resource_filename(__name__, '../src'))
from dense_deep_event_stereo import dataset_constants

BRIDGE = cv_bridge.CvBridge()

TOPICS = {
    'depth': '/davis/left/depth_image_rect',
    'cam0': {
        'image': '/davis/left/image_raw',
        'events': '/davis/left/events',
    },
    'cam1': {
        'image': '/davis/right/image_raw',
        'events': '/davis/right/events'
    }
}


def _depth2disparity(depth_image, focal_length_x_baseline):

    disparity_image = np.round(dataset_constants.DISPARITY_MULTIPLIER *
                               np.abs(focal_length_x_baseline) /
                               (depth_image + 1e-15))
    invalid = np.isnan(disparity_image) | (disparity_image == float('inf')) | (
        disparity_image >= 255.0)
    disparity_image[invalid] = dataset_constants.INVALID_DISPARITY

    return disparity_image.astype(np.uint8)


def _rectification_map(intrinsics_extrinsics):
    """Produces tables that map rectified coordinates to distorted coordinates.

       x_distorted = rectified_to_distorted_x[y_rectified, x_rectified]
       y_distorted = rectified_to_distorted_y[y_rectified, x_rectified]
    """
    dist_coeffs = intrinsics_extrinsics['distortion_coeffs']
    D = np.array(dist_coeffs)

    intrinsics = intrinsics_extrinsics['intrinsics']
    K = np.array([[intrinsics[0], 0., intrinsics[2]],
                  [0., intrinsics[1], intrinsics[3]], [0., 0., 1.]])
    K_new = np.array(intrinsics_extrinsics['projection_matrix'])[0:3, 0:3]

    R = np.array(intrinsics_extrinsics['rectification_matrix'])

    size = (intrinsics_extrinsics['resolution'][0],
            intrinsics_extrinsics['resolution'][1])

    rectified_to_distorted_x, rectified_to_distorted_y = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, K_new, size, cv2.CV_32FC1)

    return rectified_to_distorted_x, rectified_to_distorted_y


def _rectify_events(events, distorted_to_rectified, image_size):

    rectified_events = []
    width, height = image_size
    for event in events:
        timestamp, x, y, polarity = event
        x_rectified = round(distorted_to_rectified[y, x][0])
        y_rectified = round(distorted_to_rectified[y, x][1])
        if (0 <= x_rectified < width) and (0 <= y_rectified < height):
            rectified_events.append(
                [timestamp, x_rectified, y_rectified, polarity])

    return rectified_events


def _load_image_message(message):
    image, timestamp = BRIDGE.imgmsg_to_cv2(
        message.message), message.timestamp.to_sec()
    return image, timestamp


def _convert_message_to_event(message):
    return [(event.ts.to_sec(), event.x, event.y, 2 * int(event.polarity) - 1)
            for event in message]


def _get_bags_timestamps(topic_reader):
    return [bag.timestamp.to_sec() for bag in topic_reader]


def _find_first_and_last_event_bags_indices(event_bag_timestamps,
                                            start_timestamp, end_timestamp):
    number_of_bags = len(event_bag_timestamps)
    for first_bag_index in range(number_of_bags - 1, -1, -1):
        if event_bag_timestamps[first_bag_index] < start_timestamp:
            break
    # Note we search for events in bag in [first_bag_index, last_bag_index).
    for last_bag_index in range(first_bag_index, number_of_bags):
        if event_bag_timestamps[last_bag_index] > end_timestamp:
            break
    return first_bag_index, last_bag_index


def _find_image_bag_index(image_bags_timestamps, synchronization_timestamp):
    image_bag_timestamps_array = np.array(image_bags_timestamps)
    return np.argmin(
        np.abs(image_bag_timestamps_array - synchronization_timestamp))


def _get_synchronized_events(end_timestamp, event_bags_timestamps,
                             events_topic_reader):
    start_timestamp = end_timestamp - dataset_constants.TIME_BETWEEN_EXAMPLES
    (first_bag_index,
     last_bag_index) = _find_first_and_last_event_bags_indices(
         event_bags_timestamps, start_timestamp, end_timestamp)
    events = []
    for bag_index in range(first_bag_index, last_bag_index):
        bag = events_topic_reader[bag_index]
        events += _convert_message_to_event(bag.message.events)
    return [
        event for event in events if start_timestamp < event[0] < end_timestamp
    ]


def _get_synchronized_image(synchronization_timestamp, image_bags_timestamps,
                            images_topic_reader):
    bag_index = _find_image_bag_index(image_bags_timestamps,
                                      synchronization_timestamp)
    return _load_image_message(images_topic_reader[bag_index])[0]


def _make_if_does_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def _is_experiment_correctly_defined(experiment):
    experiment_name, experiment_number = experiment
    print(experiment_name, experiment_number)
    if experiment_name not in dataset_constants.EXPERIMENTS:
        return False
    if experiment_number not in dataset_constants.EXPERIMENTS[experiment_name]:
        return False
    return True


@click.command()
@click.argument('dataset_folder', type=click.Path(exists=False))
@click.option('--temporal_folder',
              default=tempfile.mkdtemp(),
              type=click.Path(exists=False))
@click.option('--experiment', type=click.Tuple([str, int]))
def main(dataset_folder, temporal_folder, experiment):
    """Run the script.

    Args:
        dataset_folder : specifies output folder
        experiment : tuple (<experiment name>, <sequence number>), that
                     specifies which experiment to download.
                     We sequences 1,2,3,4 of indoor_flying experiment.
        temporal_folder : specifies folder were the original dataset
                          will be placed.
    """
    if not _is_experiment_correctly_defined(experiment):
        raise ValueError('"experiments" are not correctly defined.')

    dataset_folder = os.path.abspath(dataset_folder)
    temporal_folder = os.path.abspath(temporal_folder)

    _make_if_does_not_exist(temporal_folder)
    _make_if_does_not_exist(dataset_folder)

    downloader.TMP_FOLDER = temporal_folder
    experiment_name, experiment_number = experiment
    paths = dataset_constants.experiment_paths(experiment_name,
                                               experiment_number,
                                               dataset_folder)
    dataset_constants.create_folders(paths)
    calibration_data = calibration.Calibration(experiment_name)

    data_path = downloader.get_data(experiment_name, experiment_number)[0]
    data_bag = bag_indexer.get_bag_indexer(data_path)

    gt_bag_path = downloader.get_ground_truth(experiment_name,
                                              experiment_number)[0]
    gt_bag = bag_indexer.get_bag_indexer(gt_bag_path)

    depth_topic_reader = gt_bag.get_topic_reader(TOPICS['depth'])
    focal_length_x_baseline = calibration_data.intrinsic_extrinsic['cam1'][
        'projection_matrix'][0][3]
    synchronization_timestamps = []
    for index, depth_message in enumerate(depth_topic_reader):
        depth_image, timestamp = _load_image_message(depth_message)
        disparity_image = _depth2disparity(depth_image,
                                           focal_length_x_baseline)
        disparity_path = paths['disparity_file'] % index
        cv2.imwrite(disparity_path, disparity_image)
        synchronization_timestamps.append(timestamp)

    np.savetxt(paths['timestamps_file'],
               np.array(synchronization_timestamps),
               fmt='%f',
               header="timestamp")

    distorted_to_rectified = {
        'cam0': calibration_data.left_map,
        'cam1': calibration_data.right_map
    }

    for camera in ['cam0', 'cam1']:
        rectified_to_distorted_x, rectified_to_distorted_y = \
         _rectification_map(
            calibration_data.intrinsic_extrinsic[camera])
        image_size = calibration_data.intrinsic_extrinsic[camera]['resolution']

        events_topic_reader = data_bag.get_topic_reader(
            TOPICS[camera]['events'])
        images_topic_reader = data_bag.get_topic_reader(
            TOPICS[camera]['image'])
        event_bags_timestamps = _get_bags_timestamps(events_topic_reader)
        image_bags_timestamps = _get_bags_timestamps(images_topic_reader)

        for synchronization_index, synchronization_timestamp in enumerate(
                synchronization_timestamps):

            synchronized_events = _get_synchronized_events(
                synchronization_timestamp, event_bags_timestamps,
                events_topic_reader)
            rectified_synchronized_events = _rectify_events(
                synchronized_events, distorted_to_rectified[camera],
                image_size)
            events_path = paths[camera]['event_file'] % synchronization_index
            np.save(events_path, np.array(rectified_synchronized_events))
            synchronized_image = _get_synchronized_image(
                synchronization_timestamp, image_bags_timestamps,
                images_topic_reader)
            rectified_synchronized_image = cv2.remap(synchronized_image,
                                                     rectified_to_distorted_x,
                                                     rectified_to_distorted_y,
                                                     cv2.INTER_LINEAR)
            image_path = paths[camera]['image_file'] % synchronization_index
            cv2.imwrite(image_path, rectified_synchronized_image)


if __name__ == '__main__':
    main()
