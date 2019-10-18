# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch as th

from dense_deep_event_stereo import events
from dense_deep_event_stereo import transformers


def test_normalize_features_to_zero_mean_unit_std():
    # yapf: disable
    features = np.array([[0.1, 0.0, 1.0, -1.0],
                         [0.8, 1.0, 1.0, 1.0],
                         [0.9, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, -1.0],
                         [1.3, 0.0, 1.0, 1.0]], dtype=np.float)
    # yapf: enable
    example = {
        'event_sequence':
        events.EventSequence(
            features,
            features_names=['timestamp', 'x', 'y', 'polarity'],
            image_height=2,
            image_width=2)
    }
    example = transformers.normalize_features_to_zero_mean_unit_std(example, 3)
    assert np.isclose(example['event_sequence']._features[:, 3].mean(), 0.0)
    assert np.isclose(example['event_sequence']._features[:, 3].std(), 1.0)


def test_event_sequence_to_event_image():
    # yapf: disable
    features = np.array([[0.9, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, -1.0],
                         [1.3, 0.0, 1.0, 1.0]], dtype=np.float)
    # yapf: enable
    example = {
        'event_sequence':
        events.EventSequence(
            features,
            features_names=['timestamp', 'x', 'y', 'polarity'],
            image_height=2,
            image_width=2)
    }
    sequence_to_image_transformer = \
        transformers.EventSequenceToEventImage()
    example = sequence_to_image_transformer(example)
    event_image = example['event_image']
    expected_event_image = th.zeros(4, 2, 2)
    # Meaning of the channels is following:
    # 0, 1 - number of positive and negative events,
    # 2, 3 - timestamp of last positive and negative events.
    expected_event_image[:, 1, 1] = th.Tensor([1, 1, 0.9, 1.0])
    expected_event_image[:, 1, 0] = th.Tensor([1, 0, 1.3, 0.0])
    assert th.isclose(expected_event_image, event_image).all()


def test_keep_recent_events_ground_truth():
    # yapf: disable
    features = np.array([[0.1, 5.0, 1.0],
                         [0.3, 1.0, 3.0],
                         [0.8, 1.0, 1.0],
                         [1.3, 5.0, 1.0]], dtype=np.float)
    # yapf: enable
    example = {
        'left': {
            'event_sequence':
            events.EventSequence(features,
                                  features_names=['timestamp', 'x', 'y'],
                                  image_height=4,
                                  image_width=6),
            'disparity_image':
            np.ones((4, 6), dtype=np.float),
        },
    }
    mask_ground_truth_transformer = \
        transformers.KeepRecentEventsGoundtruth(number_of_events=2)
    mask_ground_truth_transformer(example)
    expected_disparity_image = np.full((4, 6), float('inf'), dtype=np.float)
    expected_disparity_image[1, 5] = 1.0
    expected_disparity_image[1, 1] = 1.0
    assert np.all(
        np.isclose(example['left']['disparity_image'],
                   expected_disparity_image))
    assert len(example['left']['event_sequence']) == 4


def test_dictionary_of_numpy_arrays_to_tensors():
    dictionary_of_numpy_arrays = {
        'item_0_0': {
            'item_1_0': np.ones((3, 4))
        },
        'item_0_1': np.zeros((4, 2))
    }
    dictionary_of_tensors = \
        transformers.dictionary_of_numpy_arrays_to_tensors(
            dictionary_of_numpy_arrays)
    assert isinstance(dictionary_of_tensors['item_0_1'], th.Tensor)
    assert isinstance(dictionary_of_tensors['item_0_0']['item_1_0'], th.Tensor)


def test_absolute_time_to_relative():
    # yapf: disable
    features = np.array([[0.1, 5.0, 1.0],
                         [0.3, 1.0, 3.0],
                         [0.6, 1.0, 1.0],
                         [1.3, 5.0, 1.0]], dtype=np.float)
    # yapf: enable
    example = {
        'event_sequence':
        events.EventSequence(features,
                              features_names=['timestamp', 'x', 'y'],
                              image_height=10,
                              image_width=20)
    }
    example = transformers.absolute_time_to_relative(example)
    # yapf: disable
    expected_features = np.array([[-1.2, 5.0, 1.0],
                                  [-1.0, 1.0, 3.0],
                                  [-0.7, 1.0, 1.0],
                                  [0.0, 5.0, 1.0]], dtype=np.float)
    # yapf: enable
    assert np.all(
        np.isclose(example['event_sequence']._features,
                   expected_features,
                   atol=1e-2))


def test_apply_transformers_to_left_right():
    def mockup_transformer(example):
        example['attribute'] -= 1.0
        return example

    example = {'left': {'attribute': 10.0}, 'right': {'attribute': 3.0}}
    transformer = transformers.ApplyTransformersToLeftRight(
        [mockup_transformer])
    example = transformer(example)
    assert example['left']['attribute'] == 9.0
    assert example['right']['attribute'] == 2.0


def test_event_sequence_to_event_queue():
    # yapf: disable
    features = np.array([[0.1, 3.0, 1.0, 1.0],
                         [0.3, 3.0, 1.0, 0.0],
                         [0.6, 3.0, 1.0, 1.0],
                         [1.3, 5.0, 2.0, 1.0]], dtype=np.float)
    # yapf: enable
    example = {
        'event_sequence':
        events.EventSequence(features,
                              features_names=['timestamp', 'x', 'y'],
                              image_height=5,
                              image_width=3)
    }
    transformer = transformers.EventSequenceToEventQueue(queue_capacity=2,
                                                         image_height=3,
                                                         image_width=6)
    example = transformer(example)
    expected_queue = th.zeros(2, 2, 3, 6)
    # yapf: disable
    expected_queue[:, :, 1, 3] = th.Tensor([[0.6, 0.3],
                                           [1.0, 0.0]])
    expected_queue[:, 0, 2, 5] = th.Tensor([1.3, 1.0])
    # yapf: enable
    assert th.isclose(example['event_queue'], expected_queue, atol=1e-2).all()
