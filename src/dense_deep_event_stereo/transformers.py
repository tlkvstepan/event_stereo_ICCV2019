# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch as th

from dense_deep_event_stereo import events
from dense_deep_event_stereo import _events


def normalize_features_to_zero_mean_unit_std(example, feature_index):
    event_sequence = example['event_sequence']
    mean = event_sequence._features[:, feature_index].mean()
    std = event_sequence._features[:, feature_index].std()
    event_sequence._features[:, feature_index] = (
        event_sequence._features[:, feature_index] - mean) / (std + 1e-10)
    example['event_sequence'] = event_sequence
    return example


def normalize_polarity(example):
    return normalize_features_to_zero_mean_unit_std(example, 3)


def _find_latest_timestamp_and_number_of_events_in_each_location(
        event_sequence, polarity):
    polarity_event_sequence = event_sequence.filter_by_polarity(polarity)

    polarity_event_sequence.spatial_hashing()
    number_of_events_in_each_location = \
                            polarity_event_sequence._events_in_location
    locations_without_events = polarity_event_sequence._events_in_location == 0
    index_of_last_event_in_each_location = np.clip(
        number_of_events_in_each_location - 1, a_min=0, a_max=None)
    indices_of_latest_events = np.squeeze(
        np.take_along_axis(
            polarity_event_sequence._spatial_hash_table,
            np.expand_dims(index_of_last_event_in_each_location, axis=2),
            axis=2))
    latest_timestamp_in_each_location = np.take(
        polarity_event_sequence._features[:, events.TIMESTAMP_COLUMN],
        indices_of_latest_events,
        axis=0)
    latest_timestamp_in_each_location[locations_without_events] = 0.0
    return (latest_timestamp_in_each_location,
            number_of_events_in_each_location)


class EventSequenceToEventImage(object):
    """Converts events sequence to image."""

    def __call__(self, example):
        """Returns "example" appended "event_image" item.

        Args:
            example: dictionary containing "event_sequence" item.

        Returns:
            example: dictionary where "event_sequence" item is substituted
                     by "event_image" item. The "event_image" is an image
                    with 4 channels (channels first). The first and the
                    second channels hold number of events of positive and
                    negative polarity. The third and the fourth hold time
                    of the last update in each location.
        """
        event_sequence = example['event_sequence']
        (latest_positive_timestamp_in_each_location,
         number_of_positive_events_in_each_location
         ) = _find_latest_timestamp_and_number_of_events_in_each_location(
             event_sequence, 1)
        (latest_negative_timestamp_in_each_location,
         number_of_negative_events_in_each_location
         ) = _find_latest_timestamp_and_number_of_events_in_each_location(
             event_sequence, -1)
        example['event_image'] = th.stack([
            th.from_numpy(number_of_positive_events_in_each_location).float(),
            th.from_numpy(number_of_negative_events_in_each_location).float(),
            th.from_numpy(latest_positive_timestamp_in_each_location).float(),
            th.from_numpy(latest_negative_timestamp_in_each_location).float()
        ], 0)
        del example['event_sequence']
        return example


class KeepRecentEventsGoundtruth(object):
    """Keeps ground truth only for most recent events."""

    def __init__(self, number_of_events=15000):
        self._number_of_events = number_of_events

    def __call__(self, example):
        event_sequence = example['left']['event_sequence']
        first_index = int(max(0, len(event_sequence) - self._number_of_events))
        trimed_event_sequence = event_sequence.clone()
        trimed_event_sequence._features = trimed_event_sequence._features[
            first_index:, :]
        trimed_event_sequence.spatial_hashing()
        mask = (trimed_event_sequence._events_in_location == 0)
        example['left']['disparity_image'][mask] = float('inf')
        return example


def dictionary_of_numpy_arrays_to_tensors(example):
    """Transforms dictionary of numpy arrays to dictionary of tensors."""
    if isinstance(example, dict):
        return {
            key: dictionary_of_numpy_arrays_to_tensors(value)
            for key, value in example.items()
        }
    if isinstance(example, np.ndarray):
        return th.from_numpy(example).float()
    return example


def absolute_time_to_relative(example):
    """Transforms absolute time to time relative to the latest event."""
    end_timestamp = example['event_sequence'].end_timestamp()
    example['event_sequence']._features[:, events.TIMESTAMP_COLUMN] -= \
        end_timestamp
    return example


class ApplyTransformersToLeftRight(object):
    """Applies transformers to left and right example's items."""

    def __init__(self, transformers):
        self._transformers = transformers

    def __call__(self, example):
        for camera_name in ['left', 'right']:
            for transformer in self._transformers:
                example[camera_name] = transformer(example[camera_name])
        return example


class EventSequenceToEventQueue(object):
    """Transforms EventSequence to location-wise event queue."""

    def __init__(self, queue_capacity, image_height, image_width):
        self._queue_capacity = queue_capacity
        self._image_width = image_width
        self._image_height = image_height

    def __call__(self, example):
        """Returns events FIFO for each location.

        Args:
            example: dictionary with "event_sequence" item of type
                     EventsSequence.

        Returns:
            Substitutes "event_sequence" by "event_queue" item, which
            is pytorch tensor with indices [event_feature,
            event_order, y, x]. Events features are timestamp
            and polarity of the event.
        """
        features_indices = np.array([0, 3])
        sequences = np.expand_dims(
            np.flip(example['event_sequence']._features, axis=0), axis=0)
        sequences_lengths = np.array([len(example['event_sequence'])])
        example['event_queue'] = th.from_numpy(
            _events.compute_spatial_binning(
                sequences, sequences_lengths, features_indices,
                self._image_height, self._image_width, self._queue_capacity,
                events.Y_COLUMN, events.X_COLUMN)[0]).permute(
                    0, 4, 3, 1, 2).squeeze(0).float()
        del example['event_sequence']
        return example
