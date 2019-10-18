# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch as th

from dense_deep_event_stereo import continuous_fully_connected


def test_continuous_fully_connected_sizes():
    network = continuous_fully_connected.ContinuousFullyConnected(
    	number_of_features=2, number_of_hidden_layers=2)
    sample_cooridinates = th.rand(2, 1, 3, 4, 5)
    sample_features = th.rand(2, 1, 3, 4, 5)
    output = network(sample_cooridinates, sample_features)
    assert output.size() == (2, 2, 4, 5)
