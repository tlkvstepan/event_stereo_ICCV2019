# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch as th

from dense_deep_event_stereo import network


def test_dense_deep_event_stereo_output_sizes():
    th.manual_seed(0)
    left_event_queue = th.rand(1, 2, 7, 65, 63)
    right_event_queue = th.rand(1, 2, 7, 65, 63)
    for event_stereo_network in [
            network.DenseDeepEventStereo.default_with_temporal_convolutions(),
            network.DenseDeepEventStereo.
            default_with_continuous_fully_connected()
    ]:
        event_stereo_network.eval()
        disparity = event_stereo_network(left_event_queue, right_event_queue)
        assert disparity.size() == (1, 65, 63)
        event_stereo_network.train()
        matching_cost = event_stereo_network(left_event_queue,
                                             right_event_queue)
        assert matching_cost.size() == (1, 32, 65, 63)
    event_stereo_network = network.DenseDeepEventStereo.default_with_hand_crafted(
    )
    event_stereo_network.eval()
    left_event_image = th.rand(1, 4, 65, 63)
    right_event_image = th.rand(1, 4, 65, 63)
    disparity = event_stereo_network(left_event_image, right_event_image)
    assert disparity.size() == (1, 65, 63)
    event_stereo_network.train()
    matching_cost = event_stereo_network(left_event_image, right_event_image)
    assert matching_cost.size() == (1, 32, 65, 63)


def test_shallow_event_stereo_output_sizes():
    th.manual_seed(0)
    left_event_queue = th.rand(1, 2, 7, 65, 63)
    right_event_queue = th.rand(1, 2, 7, 65, 63)
    for event_stereo_network in [
            network.ShallowEventStereo.default_with_temporal_convolutions(),
            network.ShallowEventStereo.default_with_continuous_fully_connected()
    ]:
        event_stereo_network.eval()
        disparity = event_stereo_network(left_event_queue, right_event_queue)
        assert disparity.size() == (1, 65, 63)
        event_stereo_network.train()
        matching_cost = event_stereo_network(left_event_queue,
                                             right_event_queue)
        assert matching_cost.size() == (1, 38, 65, 63)
    event_stereo_network = network.ShallowEventStereo.default_with_hand_crafted()
    event_stereo_network.eval()
    left_event_image = th.rand(1, 4, 65, 63)
    right_event_image = th.rand(1, 4, 65, 63)
    disparity = event_stereo_network(left_event_image, right_event_image)
    assert disparity.size() == (1, 65, 63)
    event_stereo_network.train()
    matching_cost = event_stereo_network(left_event_image, right_event_image)
    assert matching_cost.size() == (1, 38, 65, 63)
