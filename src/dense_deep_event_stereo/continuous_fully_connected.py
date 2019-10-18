# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import torch as th
from torch import nn


def _convolution1x1x1(number_of_input_features, number_of_output_features):
    return nn.Conv3d(number_of_input_features,
                     number_of_output_features,
                     kernel_size=1)


def _convolution1x1x1_with_relu(number_of_input_features,
                                number_of_output_features):
    return nn.Sequential(
        _convolution1x1x1(number_of_input_features, number_of_output_features),
        nn.ReLU(inplace=True))


class KernelNetwork(nn.Module):
    def __init__(self,
                 number_of_features=64,
                 number_of_hidden_layers=2,
                 custom_initialization=True):
        super(KernelNetwork, self).__init__()
        self._number_of_features = number_of_features
        kernel_network = [_convolution1x1x1_with_relu(1, number_of_features)]
        kernel_network += [
            _convolution1x1x1_with_relu(number_of_features, number_of_features)
            for _ in range(number_of_hidden_layers)
        ]
        kernel_network.append(
            _convolution1x1x1(number_of_features, number_of_features))
        self._kernel_network = nn.ModuleList(kernel_network)
        if custom_initialization:
            self._initialize()

    def forward(self, coordinates):
        output = coordinates
        for _module in self._kernel_network:
            output = _module(output)
        return output

    def _initialize(self,
                    minimum_coordinate=-1.0,
                    maximum_coordinate=0.0,
                    number_of_iterations=5000,
                    number_of_ties=100):
        criterion = th.nn.MSELoss()
        optimizer = th.optim.Adam(self.parameters())
        random_target = th.randn(1, self._number_of_features, number_of_ties,
                                 1, 1)
        random_target /= math.sqrt(self._number_of_features)
        network_input = th.linspace(minimum_coordinate, maximum_coordinate,
                                    number_of_ties).view(
                                        1, 1, number_of_ties, 1, 1)
        for _ in range(number_of_iterations):
            optimizer.zero_grad()
            network_output = self(network_input)
            loss = criterion(network_output, random_target)
            loss.backward()
            optimizer.step()


class ContinuousFullyConnected(nn.Module):
    def __init__(self, number_of_features, number_of_hidden_layers):
        super(ContinuousFullyConnected, self).__init__()

        self._kernel_network = KernelNetwork(number_of_features,
                                             number_of_hidden_layers)
        # Last layer produces weights, and thus does not have nonlinerarity.
        self._biases = nn.Parameter(th.zeros(1, number_of_features, 1, 1))

    def forward(self, coordinates, features):
        """Returns output of forward pass.
        Args:
            coordinates: tensor with coordinates with
                         indices [batch_index, 1, event_index,
                         y, x].
            features: tensor with features that correspond to
                      the "sample_coordinates" with indices
                      [batch_index, 1, event_index, y, x]
        The locations with zero features are assumed to be empty.
        """
        # Compute kernel weights for our samples.
        weights = self._kernel_network(coordinates)
        nonzero_features = features != 0.0
        # Weights are of size (batch_size, number_of_output_features,
        # number_of_events, height, width). Use broadcasting.
        return th.sum(weights * features, dim=2) / (
            nonzero_features.sum(dim=2).float() + 1e-10) + self._biases
