# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil

import numpy as np

from dense_deep_event_stereo import dataset
from dense_deep_event_stereo import events
from test import dataset_mockup


def check_example(example):
    assert isinstance(example['left']['event_sequence'],
                      events.EventSequence)
    assert isinstance(example['right']['event_sequence'],
                      events.EventSequence)
    assert example['left']['disparity_image'].shape == (260, 346)
    assert 'timestamp' in example
    assert 'image' in example['left']


def test_dataset():
    dataset_folder = dataset_mockup.create_mvsec_dataset_mockup()
    train_set = dataset.MvsecDataset.dataset(
        dataset_folder,
        experiments=dataset_mockup.MOCKUP_MVSEC_STEREO_EXPERIMENT)
    assert len(train_set) > 0
    check_example(train_set[0])

    # Check split
    first_subset, second_subset = train_set.split_into_two(2)
    assert len(first_subset) == 2
    assert len(second_subset) == len(train_set) - 2

    # Check accumulation time modification.
    time_horizon = 0.25
    train_set.set_time_horizon(time_horizon)
    assert np.isclose(
        train_set[6]['left']['event_sequence'].duration(),
        time_horizon,
        atol=1e-1)

    # Check number of events modification.
    print(len(train_set[2]['left']['event_sequence']))
    print(len(train_set[4]['right']['event_sequence']))
    train_set.set_number_of_events(2)
    print(len(train_set[2]['left']['event_sequence']))
    print(len(train_set[4]['right']['event_sequence']))
    assert len(train_set[2]['left']['event_sequence']) == 2
    assert len(train_set[4]['right']['event_sequence']) == 2

    # Check subsampling.
    train_set.subsample(2)
    assert len(train_set) == 2
    for example_index, _ in enumerate(train_set):
        pass
    assert example_index == 1

    shutil.rmtree(dataset_folder)
