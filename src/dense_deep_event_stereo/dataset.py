# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import PIL.Image
import numpy as np
import os
import random
from functools import reduce

from dense_deep_event_stereo import dataset_constants
from dense_deep_event_stereo import events

# For test we use same frames as
# "Realtime Time Synchronized Event-based Stereo"
# by Alex Zhu et al. for consistency of test results.
FRAMES_FILTER_FOR_TEST = {
    'indoor_flying': {
        1: list(range(140, 1201)),
        2: list(range(120, 1421)),
        3: list(range(73, 1616)),
        4: list(range(190, 290))
    }
}

# For the training we use different frames, since we found
# that frames recomended by "Realtime Time Synchronized
# Event-based Stereo" by Alex Zhu include some still frames.
FRAMES_FILTER_FOR_TRAINING = {
    'indoor_flying': {
        1: list(range(80, 1260)),
        2: list(range(160, 1580)),
        3: list(range(125, 1815)),
        4: list(range(190, 290))
    }
}


def _filter_examples(examples, frames_filter):
    return [
        example for example in examples
        if frames_filter[example['experiment_name']][
            example['experiment_number']] is None or example['frame_index'] in
        frames_filter[example['experiment_name']][example['experiment_number']]
    ]


def _get_examples_from_experiments(experiments, dataset_folder):
    examples = []
    for experiment_name, experiment_numbers in experiments.items():
        for experiment_number in experiment_numbers:
            examples += _get_examples_from_experiment(experiment_name,
                                                      experiment_number,
                                                      dataset_folder)
    return examples


def _get_examples_from_experiment(experiment_name, experiment_number,
                                  dataset_folder):
    examples = []
    paths = dataset_constants.experiment_paths(experiment_name,
                                               experiment_number,
                                               dataset_folder)
    timestamps = np.loadtxt(paths['timestamps_file'])
    frames_number = timestamps.shape[0]

    for frame_index in range(frames_number):
        example = {}
        example['experiment_name'] = experiment_name
        example['experiment_number'] = experiment_number
        example['frame_index'] = frame_index
        example['timestamp'] = timestamps[frame_index]
        example['left_image_path'] = paths['cam0']['image_file'] % frame_index
        example['disparity_image_path'] = paths['disparity_file'] % frame_index
        examples.append(example)

    return examples


def _get_image(image_path):
    # Not all examples have images.
    if not os.path.isfile(image_path):
        return np.zeros(
            (dataset_constants.IMAGE_HEIGHT, dataset_constants.IMAGE_WIDTH),
            dtype=np.uint8)
    return np.array(PIL.Image.open(image_path)).astype(np.uint8)


def _get_disparity_image(disparity_image_path):
    disparity_image = _get_image(disparity_image_path)
    invalid_disparity = (
        disparity_image == dataset_constants.INVALID_DISPARITY)
    disparity_image = (disparity_image /
                       dataset_constants.DISPARITY_MULTIPLIER)
    disparity_image[invalid_disparity] = float('inf')
    return disparity_image


class MvsecDataset():
    def __init__(self, examples, dataset_folder, transformers_list=None):
        self._examples = examples
        self._transformers = transformers_list
        self._dataset_folder = dataset_folder
        self._events_time_horizon = dataset_constants.TIME_BETWEEN_EXAMPLES
        self._number_of_events = float('inf')

    def set_time_horizon(self, time):
        self._events_time_horizon = time
        self._number_of_events = float('inf')

    def set_number_of_events(self, number_of_events):
        self._events_time_horizon = float('inf')
        self._number_of_events = number_of_events

    def _accumulate_events(self, experiment_name, experiment_number,
                           frame_index):
        # In the dataset all examples are sequential video frames.
        # Therefore, to accumulate events over longer time than
        # standard 0.05 sec one just need to consider examples with
        # smaller indices than the index of the current example.
        paths = dataset_constants.experiment_paths(experiment_name,
                                                   experiment_number,
                                                   self._dataset_folder)
        first_index = int(
            max(-1, (frame_index -
                     np.ceil(self._events_time_horizon /
                             dataset_constants.TIME_BETWEEN_EXAMPLES))))
        left_event_sequences, right_event_sequences = [], []
        number_of_events = 0
        for previous_frame_index in range(frame_index, first_index, -1):
            left_events_filename = paths['cam0'][
                'event_file'] % previous_frame_index
            right_events_filename = paths['cam1'][
                'event_file'] % previous_frame_index
            event_sequence = events.EventSequence.from_npy(
                left_events_filename,
                camera_name='left',
                image_width=dataset_constants.IMAGE_WIDTH,
                image_height=dataset_constants.IMAGE_HEIGHT)
            number_of_events += len(event_sequence)
            left_event_sequences.append(event_sequence)
            right_event_sequences.append(
                events.EventSequence.from_npy(
                    right_events_filename,
                    camera_name='right',
                    image_width=dataset_constants.IMAGE_WIDTH,
                    image_height=dataset_constants.IMAGE_HEIGHT))
            if number_of_events > self._number_of_events:
                break
        left_event_sequence = reduce(
            (lambda element1, element2: element1 + element2),
            left_event_sequences)
        right_event_sequence = reduce(
            (lambda element1, element2: element1 + element2),
            right_event_sequences)
        left_event_sequence._features = left_event_sequence._features[int(
            max(0,
                len(left_event_sequence) -
                self._number_of_events)):, :].copy()
        right_event_sequence._features = right_event_sequence._features[int(
            max(0,
                len(right_event_sequence) -
                self._number_of_events)):, :].copy()
        return left_event_sequence, right_event_sequence

    def split_into_two(self, first_subset_size):
        return (self.__class__(self._examples[:first_subset_size],
                               self._dataset_folder,
                               transformers_list=self._transformers),
                self.__class__(self._examples[first_subset_size:],
                               self._dataset_folder,
                               transformers_list=self._transformers))

    def shuffle(self, random_seed=0):
        """Shuffle examples in the dataset.

        By setting "random_seed", one can ensure that order will be the
        same across different runs. This is usefull for visualization of
        examples during the traininig.
        """
        random.seed(random_seed)
        random.shuffle(self._examples)

    def subsample(self, number_of_examples, random_seed=0):
        """Keeps "number_of_examples" examples in the dataset.

        By setting "random_seed", one can ensure that subset of examples
        will be same in a different runs. This method is usefull for
        debugging.
        """
        random.seed(random_seed)
        self._examples = random.sample(self._examples, number_of_examples)

    def __len__(self):
        return len(self._examples)

    def get_example(self, index):
        example = self._examples[index]
        (left_event_sequence,
         right_event_sequence) = self._accumulate_events(
             example['experiment_name'], example['experiment_number'],
             example['frame_index'])
        return {
            'left': {
                'image':
                _get_image(example['left_image_path']),
                'event_sequence':
                left_event_sequence,
                'disparity_image':
                _get_disparity_image(example['disparity_image_path']),
            },
            'right': {
                'event_sequence': right_event_sequence
            },
            'timestamp': example['timestamp']
        }

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        example = self.get_example(index)
        if self._transformers is not None:
            for transformer in self._transformers:
                example = transformer(example)
        return example

    @staticmethod
    def disparity_to_depth(disparity_image):
        """Converts disparity to depth."""
        raise NotImplementedError('"disparity_to_depth" method should '
                                  'be implemented in a child class.')

    @classmethod
    def dataset(cls, dataset_folder, experiments, frames_filter=None):
        examples = _get_examples_from_experiments(experiments, dataset_folder)
        if frames_filter is not None:
            examples = _filter_examples(examples, frames_filter)
        return cls(examples, dataset_folder)


class IndoorFlying(MvsecDataset):
    @staticmethod
    def disparity_to_depth(disparity_image):
        unknown_disparity = disparity_image == float('inf')
        depth_image = \
            dataset_constants.FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (
            disparity_image + 1e-7)
        depth_image[unknown_disparity] = float('inf')
        return depth_image

    @staticmethod
    def split(dataset_folder, split_number):
        """Creates training, validation and test sets.

        Args:
            dataset_folder: path to dataset.
            split_number: number of split (same as number of test sequence).
        """
        if split_number == 1:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [1]},
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle()
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [2, 3]},
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        elif split_number == 2:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [2]},
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle()
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [1, 3]},
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        elif split_number == 3:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [3]},
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle()
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [1, 2]},
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        else:
            raise ValueError('Test sequence should be equal to 1, 2 or 3.')
