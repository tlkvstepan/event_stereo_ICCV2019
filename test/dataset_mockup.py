# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import numpy as np
import PIL.Image

from dense_deep_event_stereo import dataset_constants

MOCKUP_MVSEC_STEREO_NUMBER_OF_EXAMPLES = 10
MOCKUP_MVSEC_STEREO_EXPERIMENT_NAME = 'indoor_flying'
MOCKUP_MVSEC_STEREO_EXPERIMENT_NUMBER = 1
MOCKUP_MVSEC_STEREO_EXPERIMENT = {
    MOCKUP_MVSEC_STEREO_EXPERIMENT_NAME:
    [MOCKUP_MVSEC_STEREO_EXPERIMENT_NUMBER]
}
MOCKUP_TRIPLETS_HALF_PATCH_SIZE = 5
MOCKUP_TRIPLETS_NUMBER_OF_EXAMPLES = 5
MOCKUP_TRIPLETS_TRIPLETS_PER_EXAMPLE = 10
MOCKUP_TRIPLETS_SEQUENCE_LENGTH = 20


def _sample_events(start_timestamp, finish_timestamp, min_x, max_x, min_y,
                   max_y, number_of_events):
    timestamp = np.random.uniform(
        start_timestamp, finish_timestamp, size=(number_of_events))
    timestamp.sort()
    return np.stack(
        [
            timestamp,
            np.random.randint(min_x, max_x + 1, size=(number_of_events)),
            np.random.randint(min_y, max_y + 1, size=(number_of_events)),
            np.random.choice(np.array([-1, 1]), size=(number_of_events))
        ],
        axis=-1)


def _sample_stereo_events(start_timestamp, finish_timestamp, disparity,
                          number_of_events):
    return _sample_events(
        start_timestamp=start_timestamp,
        finish_timestamp=finish_timestamp,
        min_x=disparity,
        max_x=dataset_constants.IMAGE_WIDTH-1,
        min_y=0,
        max_y=dataset_constants.IMAGE_HEIGHT-1,
        number_of_events=number_of_events)


def _computer_right_camera_events(left_camera_events, disparity):
    right_camera_events = np.copy(left_camera_events)
    right_camera_events[:, 1] -= disparity
    return right_camera_events


def _save_events(filename, events):
    np.save(
        filename,
        events)


def _sample_disparity():
    return np.random.randint(0, dataset_constants.DISPARITY_MAXIMUM)


def _make_disparity_image(disparity):
    disparity_image = np.full(
        (dataset_constants.IMAGE_HEIGHT, dataset_constants.IMAGE_WIDTH),
        disparity * dataset_constants.DISPARITY_MULTIPLIER,
        dtype=np.int)
    disparity_image = np.clip(disparity_image, a_min=None, a_max=254)
    invalid_mask = np.random.rand(dataset_constants.IMAGE_HEIGHT,
                                  dataset_constants.IMAGE_WIDTH) < 0.5
    disparity_image[invalid_mask] = dataset_constants.INVALID_DISPARITY
    return disparity_image


def _save_disparity_image(filename, disparity_image):
    PIL.Image.fromarray(disparity_image.astype(np.uint8)).save(filename)


def _add_xy_noise(events):
    events_with_noise = np.copy(events)
    noise = np.random.randint(-1, 2, size=(events.shape[0], 2))
    events_with_noise[:, 1:3] += noise
    events_with_noise[:, 1:3] = np.clip(
        events_with_noise[:, 1:3], a_min=-5, a_max=5)
    return events_with_noise


def create_mvsec_dataset_mockup():
    """Creates and returns path to mvsec stereo dataset mockup.

    Creates mockup for 6 examples from "indoor_flying1" dataset.
    The mockup dataset has the same folders structure and files
    format as the "real" mvsec stereo dataset.

    The disparity image is simulated as following:
    1. same disparity is set for for all (x, y) locations;
    2. for each example the disparity is sampled from the uniform
       distribution U([0, maximum_disparity]).
    3. for half of all location (x, y) in each example the disparity
       is set to unknown.

    The events for the left camera are simulated as following:
    1. number of events is set equal to the half of number of locations.
    2. events timestamps are sampled from the uniform distribution
       U([start_timestamp, end_timestamp]), where
       start_timestamp = example_index * time_between_examples
       finish_timestamp = start_timestamp + events_accumulation_time.
    3. events coordinates (y, x ) are sampled from the uniformed distribution
       U([0, height] x [0, width - disparity]).

    The y-position and the timestamps of the events from the right camera
    are similar to the y-positions and timestamps of the corresponding events
    from the left camera. The x-position of the right
    camera events is shifted left by disparity with respect to the x-position
    of the corresponding events from the left camera.

    Note, that users should manually delete the mockup dataset after, when it
    is no longer needed.
    """
    np.random.seed(0)
    dataset_rootpath = tempfile.mkdtemp()
    number_of_events = int(
        np.round(0.5 * dataset_constants.IMAGE_WIDTH *
                 dataset_constants.IMAGE_HEIGHT))
    paths = dataset_constants.experiment_paths(
        dataset_root=dataset_rootpath,
        experiment_name=MOCKUP_MVSEC_STEREO_EXPERIMENT_NAME,
        experiment_number=MOCKUP_MVSEC_STEREO_EXPERIMENT_NUMBER)
    dataset_constants.create_folders(paths)
    timestamps = []
    for example_index in range(0, MOCKUP_MVSEC_STEREO_NUMBER_OF_EXAMPLES):
        disparity_filename = paths['disparity_file'] % example_index
        left_camera_events_filename = (
            paths['cam0']['event_file'] % example_index)
        right_camera_events_filename = (
            paths['cam1']['event_file'] % example_index)

        disparity = _sample_disparity()
        disparity_image = _make_disparity_image(disparity)

        start_timestamp = (
            example_index * dataset_constants.TIME_BETWEEN_EXAMPLES)
        finish_timestamp = (
            dataset_constants.TIME_BETWEEN_EXAMPLES + start_timestamp)
        left_events = _sample_stereo_events(start_timestamp, finish_timestamp,
                                            disparity, number_of_events)
        right_events = _computer_right_camera_events(left_events, disparity)
        _save_disparity_image(disparity_filename, disparity_image)
        _save_events(left_camera_events_filename, left_events)
        _save_events(right_camera_events_filename, right_events)
        timestamps.append(finish_timestamp)

    np.savetxt(
        paths['timestamps_file'],
        np.array(timestamps),
        fmt='%f',
        header="timestamp")
    return dataset_rootpath


def create_triplets_dataset_mockup():
    """Creates and returns path to triplets dataset mockup.

    Creates triplets dataset mockup for 5 examples from
    "indoor_flying1" dataset. For each example, 10 triplets are
    created.

    The mockup dataset has the same folders structure and the files
    format as the "real" triplets dataset.

    The reference, match and non-match event sequences are simulated
    as following:
    1. number of events in all sequences is equal to 20.
    2. events coordinates (y, x) are sampled from the uniformed distribution
       U([-half_patch_size, half_patch_size] x
         [-half_patch_size, half_patch_size]).
    3. events timestamps are sampled from the uniform distribution
       U([start_timestamp, end_timestamp]), where
       start_timestamp = example_index * time_between_examples
       finish_timestamp = start_timestamp + events_accumulation_time.
    4. The match sequence is a copy of the reference sequence with the
       noise added to spatial coordinates of the events and the non-match
       sequence is created independently

    Note, that users should manually delete the mockup dataset, when it is no
    longer needed.
    """
    np.random.seed(0)
    dataset_rootpath = tempfile.mkdtemp()
    filename_pattern = os.path.join(dataset_rootpath, '%0.6i')
    for example_index in range(0, MOCKUP_TRIPLETS_NUMBER_OF_EXAMPLES):
        start_timestamp = (
            example_index * dataset_constants.TIME_BETWEEN_EXAMPLES)
        finish_timestamp = (
            dataset_constants.TIME_BETWEEN_EXAMPLES + start_timestamp)
        (reference_sequences_list, match_sequences_list,
         nomatch_sequences_list) = [], [], []
        for triplet_index in range(0, MOCKUP_TRIPLETS_TRIPLETS_PER_EXAMPLE):
            reference_sequence = _sample_events(
                start_timestamp=start_timestamp,
                finish_timestamp=finish_timestamp,
                min_x=-MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                max_x=MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                min_y=-MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                max_y=MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                number_of_events=MOCKUP_TRIPLETS_SEQUENCE_LENGTH)
            reference_sequences_list.append(reference_sequence)
            match_sequences_list.append(_add_xy_noise(reference_sequence))
            nomatch_sequences_list.append(
                _sample_events(
                    start_timestamp=start_timestamp,
                    finish_timestamp=finish_timestamp,
                    min_x=-MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                    max_x=MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                    min_y=-MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                    max_y=MOCKUP_TRIPLETS_HALF_PATCH_SIZE,
                    number_of_events=MOCKUP_TRIPLETS_SEQUENCE_LENGTH))

        reference_sequences = np.stack(reference_sequences_list, 0)
        nomatch_sequences = np.stack(nomatch_sequences_list, 0)
        event_sequence_lengths = np.full(
            (MOCKUP_TRIPLETS_TRIPLETS_PER_EXAMPLE),
            MOCKUP_TRIPLETS_SEQUENCE_LENGTH)
        np.savez_compressed(
            filename_pattern % example_index,
            reference_sequences=reference_sequences,
            reference_sequences_lengths=event_sequence_lengths,
            match_sequences=reference_sequences,
            match_sequences_lengths=event_sequence_lengths,
            nomatch_sequences=nomatch_sequences,
            nomatch_sequences_lengths=event_sequence_lengths)
    return dataset_rootpath
