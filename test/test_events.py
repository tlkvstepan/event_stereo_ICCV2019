# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile

import numpy as np

from dense_deep_event_stereo import _events
from dense_deep_event_stereo import events


def create_mockup_event_sequence():
    """Creates mockup events sequence."""
    np.random.seed(0)
    timestamp = np.array([0.5, 0.6, 0.2, 0.3, 0.1])
    polarity = np.array([-1, 1, 1, 1, -1])
    x = np.array([50, 200, 200, 300, 150])
    y = np.array([100, 50, 50, 0, 150])
    camera = np.array([1, 1, 0, 0, 0])
    disparity = np.full((5), float('inf'))

    features = np.stack([timestamp, x, y, polarity, camera, disparity], axis=1)
    features_names = [
        'timestamp', 'x', 'y', 'polarity', 'camera_id', 'disparity'
    ]
    event_sequence = events.EventSequence(features,
                                          features_names,
                                          image_height=260,
                                          image_width=346)
    event_sequence.sort_by_timestamp()
    return event_sequence


def test_compute_spatial_binning():
    sequence = np.array(
        [[0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 4], [1, 0, 5], [0, 0, 0]],
        dtype=np.float).reshape((1, 6, 3))
    sequence_length = np.array([5], dtype=np.long)
    features_indices = np.array([0, 2], dtype=np.long)
    height = 3
    width = 2
    maximum_local_sequence_length = 2
    y_column = 1
    x_column = 0

    local_sequences, local_sequences_lenghts = _events.compute_spatial_binning(
        sequence, sequence_length, features_indices, height, width,
        maximum_local_sequence_length, y_column, x_column)

    expected_local_sequences_lenghts = np.zeros((height, width), dtype=np.long)
    expected_local_sequences_lenghts[0, 0] = 2
    expected_local_sequences_lenghts[0, 1] = 2
    assert np.all(expected_local_sequences_lenghts == local_sequences_lenghts)
    expected_local_sequences = np.zeros(
        (height, width, maximum_local_sequence_length, 2))
    expected_local_sequences[0, 0, 0, :] = np.array([0, 1])
    expected_local_sequences[0, 0, 1, :] = np.array([0, 2])
    expected_local_sequences[0, 1, 0, :] = np.array([1, 4])
    expected_local_sequences[0, 1, 1, :] = np.array([1, 5])
    assert np.all(expected_local_sequences == local_sequences)


def test_event_sequence():
    file_with_events = tempfile.mkstemp(suffix='.npy')[1]
    np.save(file_with_events, np.array([
        [0.1, 10, 10, -1],
        [0.2, 30, 100, 1],
    ]))
    sequence_from_file = events.EventSequence.from_npy(file_with_events,
                                                       camera_name='left',
                                                       image_height=260,
                                                       image_width=346)
    assert len(sequence_from_file) == 2
    assert {
        'timestamp': 0.1,
        'x': 10.0,
        'y': 10.0,
        'polarity': -1.0,
        'camera_id': 0.0,
        'disparity': float('inf')
    } == sequence_from_file[0]

    sequence = create_mockup_event_sequence()

    assert np.all(
        sequence._features[:, 0] == np.array([0.1, 0.2, 0.3, 0.5, 0.6]))
    assert len(sequence) == 5
    assert sequence.start_timestamp() == 0.1
    assert sequence.end_timestamp() == 0.6
    assert {
        'timestamp': 0.1,
        'x': 150.0,
        'y': 150.0,
        'polarity': -1.0,
        'camera_id': 0.0,
        'disparity': float('inf')
    } == sequence[0]

    sequence.set_disparity(0, 10)
    assert {
        'timestamp': 0.1,
        'x': 150.0,
        'y': 150.0,
        'polarity': -1.0,
        'camera_id': 0.0,
        'disparity': 10.0
    } == sequence[0]

    image = sequence.to_image()
    assert np.all(image[:, 0, 0] == np.array([[255, 255, 255]]))
    assert np.all(image[:, 100, 50] == np.array([[0, 0, 255]]))
    assert np.all(image[:, 50, 200] == np.array([[255, 0, 0]]))

    disparity_image = sequence.to_disparity_image()
    assert disparity_image[0, 0] == 255
    assert disparity_image[150, 150] == 7 * 10

    negative_event_sequence = sequence.filter_by_polarity(-1)
    assert not (
        negative_event_sequence.
        _features[:, negative_event_sequence.get_column_index('polarity')] == 1
    ).any()

    sequence0 = sequence.filter_by_camera_name('left')
    sequence1 = sequence.filter_by_camera_name('right')
    assert len(sequence0) == 3
    assert len(sequence1) == 2
    assert len(sequence) == len(sequence0 + sequence1)

    sequence_half = sequence.filter_by_timestamp(0.0, 0.4)
    assert len(sequence_half) == 3

    sequence_clone = sequence.clone()
    assert np.all(sequence_clone._features == sequence._features)

    sequence.spatial_hashing()
    assert sequence._spatial_hash_table is not None
    assert sequence._events_in_location is not None

    # Check if the copy is deep.
    sequence_copy = sequence.clone()
    sequence_copy._features_names += ['embedding']
    sequence_copy._features[0, 0] = 1000
    assert len(sequence._features_names) == 6
    assert len(sequence._features) != 1000


def test_compute_maximum_events_in_location():
    event_sequence = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.3, 1, 0]])
    y_column = 2
    x_column = 1
    image_height = 2
    image_width = 2

    assert 2 == _events.compute_maximum_events_in_location(
        event_sequence, y_column, x_column, image_height, image_width)


def test_compute_spatial_hash_table():
    event_sequence = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.3, 1, 0]],
                              dtype=np.float)
    y_column = 2
    x_column = 1
    image_height = 2
    image_width = 2

    expected_spatial_hash_table = np.zeros((image_height, image_height, 2),
                                           dtype=np.int)
    expected_spatial_hash_table[0, 0, 0] = 0
    expected_spatial_hash_table[0, 0, 1] = 1
    expected_spatial_hash_table[0, 1, 0] = 2

    expected_events_in_location = np.array([[2, 1], [0, 0]], dtype=np.int)

    actual_spatial_hash_table, actual_events_in_location = \
        _events.compute_spatial_hash_table(event_sequence, y_column,
                                           x_column, image_height, image_width)
    assert np.all(actual_spatial_hash_table == expected_spatial_hash_table)
    assert np.all(actual_events_in_location == expected_events_in_location)
