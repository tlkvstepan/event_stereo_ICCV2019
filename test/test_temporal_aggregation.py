# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch as th

from dense_deep_event_stereo import temporal_aggregation


def test_temporal_convolutional_output_sizes():
    th.manual_seed(0)
    events_fifo = th.rand(3, 2, 7, 4, 5)
    projection_module = temporal_aggregation.TemporalConvolutional()
    events_projection = projection_module(events_fifo)
    assert events_projection.size() == (3, 64, 4, 5)


def test_continuous_fully_connected_output_size():
    network = temporal_aggregation.ContinuousFullyConnected(64, 1)
    events_fifo = th.rand(3, 2, 10, 4, 5)
    events_projection = network(events_fifo)
    assert events_projection.size() == (3, 64, 4, 5)
