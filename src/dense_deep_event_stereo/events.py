# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import torch as th

from dense_deep_event_stereo import _events
from practical_deep_stereo import visualization

TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2

CAMERA_ID_TO_NAME = ['left', 'right']


def camera_name_to_id(camera_name):
    if camera_name not in CAMERA_ID_TO_NAME:
        raise ValueError('"camera_name" should be "left" or "right".')
    return CAMERA_ID_TO_NAME.index(camera_name)


def camera_id_to_name(camera_id):
    return CAMERA_ID_TO_NAME[int(camera_id)]


class EventSequence(object):
    """Stores events as a time sequence, simulating stream of events.

    Inside the object events are sorted in oldest-first order. All
    features of the events are stored in numpy array "features".
    The names of the features are stored separatly in "features_names"
    list as strings. The timestamp of the event is stored in first column,
    x in the second colum and y in the third column
    """
    def __init__(self, features, features_names, image_height, image_width):
        if features_names[TIMESTAMP_COLUMN] != "timestamp":
            raise ValueError('First column should store timestamp.')
        if features_names[X_COLUMN] != "x":
            raise ValueError('Second column should store x coordinate.')
        if features_names[Y_COLUMN] != "y":
            raise ValueError('Third column should store y coordinate.')

        self._events_in_location = None
        self._spatial_hash_table = None
        self._features = np.copy(features)
        self._image_height = image_height
        self._image_width = image_width
        self._features_names = copy.deepcopy(features_names)
        if not self.is_sorted():
            self.sort_by_timestamp()

    def is_sorted(self):
        return np.all(self._features[:-1, TIMESTAMP_COLUMN] <=
                      self._features[1:, TIMESTAMP_COLUMN])

    def __len__(self):
        return self._features.shape[0]

    def __getitem__(self, index):
        return {
            key: self._features[index, column]
            for column, key in enumerate(self._features_names)
        }

    def set_disparity(self, index, disparity):
        self._features[index, self.get_column_index('disparity')] = disparity

    def get_column_index(self, name):
        return self._features_names.index(name)

    def sort_by_timestamp(self):
        if len(self._features[:, TIMESTAMP_COLUMN]) > 0:
            sort_indices = np.argsort(self._features[:, TIMESTAMP_COLUMN])
            self._features = self._features[sort_indices]

    def duration(self):
        return self.end_timestamp() - self.start_timestamp()

    def start_timestamp(self):
        return self._features[0, TIMESTAMP_COLUMN]

    def end_timestamp(self):
        return self._features[-1, TIMESTAMP_COLUMN]

    def filter_by_polarity(self, polarity, make_deep_copy=True):
        polarity_column = self.get_column_index('polarity')
        mask = self._features[:, polarity_column] == polarity
        return self.filter_by_mask(mask, make_deep_copy)

    def filter_by_mask(self, mask, make_deep_copy=True):
        if not np.any(mask):
            raise ValueError('should be at least one event sub-sequence.')
        if make_deep_copy:
            return EventSequence(features=np.copy(self._features[mask]),
                                  features_names=copy.deepcopy(
                                      self._features_names),
                                  image_height=self._image_height,
                                  image_width=self._image_width)
        else:
            return EventSequence(features=self._features[mask],
                                  features_names=self._features_names,
                                  image_height=self._image_height,
                                  image_width=self._image_width)

    def filter_by_timestamp(self, start_time, duration, make_deep_copy=True):
        end_time = start_time + duration
        mask = (start_time < self._features[:, TIMESTAMP_COLUMN]) & (
            end_time >= self._features[:, TIMESTAMP_COLUMN])
        return self.filter_by_mask(mask, make_deep_copy)

    def filter_by_camera_name(self, camera_name, make_deep_copy=True):
        return self.filter_by_mask(
            self._features[:, self.get_column_index('camera_id')].astype(
                np.int32) == camera_name_to_id(camera_name), make_deep_copy)

    def to_disparity_image(self, disparity_multiplier=7.0):
        """Visualizes disparity of the events as an image.

        The pixels are colored with shades of gray depending
        on the disparity (disparities are multiplied by
        "disparity_multiplier"). If pixel does not recieve
        events or disparity of all its events is unknown it is
        shown as white.
        """
        disparity_image = np.full((self._image_height, self._image_width), 255)
        with_disparity_mask = self._features[:,
                                             self.get_column_index(
                                                 'disparity')] != float('inf')
        x = self._features[with_disparity_mask,
                           self.get_column_index('x')].astype(np.int32)
        y = self._features[with_disparity_mask,
                           self.get_column_index('y')].astype(np.int32)
        disparity = self._features[with_disparity_mask,
                                   self.get_column_index('disparity')]
        disparity_image[y, x] = disparity * disparity_multiplier
        return disparity_image.astype(np.uint8)

    def spatial_hashing(self):
        """Adds spatial hash table to the event sequence."""
        self._spatial_hash_table, self._events_in_location = \
            _events.compute_spatial_hash_table(
                self._features, Y_COLUMN, X_COLUMN,
                self._image_height, self._image_width)

    def to_image(self, background=None):
        """Visualizes stream of event as an image.

        The pixel is shown as red if dominant polarity of pixel's
        events is 1, as blue if dominant polarity of pixel's
        events is -1 and white if pixel does not recieve any events,
        or it's events does not have dominant polarity.

        Args:
            background: is (channel x height x width) image.
        """
        polarity = self._features[:, self.get_column_index('polarity')] == -1.0
        x_negative = self._features[~polarity, X_COLUMN].astype(np.int)
        y_negative = self._features[~polarity, Y_COLUMN].astype(np.int)
        x_positive = self._features[polarity, X_COLUMN].astype(np.int)
        y_positive = self._features[polarity, Y_COLUMN].astype(np.int)

        positive_histogram, _, _ = np.histogram2d(
            x_positive,
            y_positive,
            bins=(self._image_width, self._image_height),
            range=[[0, self._image_width], [0, self._image_height]])
        negative_histogram, _, _ = np.histogram2d(
            x_negative,
            y_negative,
            bins=(self._image_width, self._image_height),
            range=[[0, self._image_width], [0, self._image_height]])

        red = np.transpose(positive_histogram < negative_histogram)
        blue = np.transpose(positive_histogram > negative_histogram)
        if background is None:
            height, width = red.shape
            background = th.full((3, height, width), 255).byte()
        else:
            background = th.from_numpy(background)
        points_on_background = visualization.plot_points_on_background(
            th.nonzero(th.from_numpy(red.astype(np.uint8))), background,
            [255, 0, 0])
        points_on_background = visualization.plot_points_on_background(
            th.nonzero(th.from_numpy(blue.astype(np.uint8))),
            points_on_background, [0, 0, 255])
        return points_on_background.numpy()

    def clone(self):
        return EventSequence(features=np.copy(self._features),
                              features_names=copy.deepcopy(
                                  self._features_names),
                              image_height=self._image_height,
                              image_width=self._image_width)

    def __add__(self, sequence):
        event_sequence = EventSequence(features=np.concatenate(
            [self._features, sequence._features]),
                                         features_names=self._features_names,
                                         image_height=self._image_height,
                                         image_width=self._image_width)
        return event_sequence


    @staticmethod
    def from_npy(path, camera_name, image_height, image_width):
        features = np.load(path)
        features = np.concatenate([
            features,
            np.full((features.shape[0], 1),
                    camera_name_to_id(camera_name),
                    dtype=np.float64),
            np.full((features.shape[0], 1), float('inf'))
        ],
                                  axis=1)
        features_names = [
            'timestamp', 'x', 'y', 'polarity', 'camera_id', 'disparity'
        ]

        return EventSequence(features, features_names, image_height,
                              image_width)
