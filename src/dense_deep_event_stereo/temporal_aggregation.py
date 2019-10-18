# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from torch import nn
from torch.nn import functional

from dense_deep_event_stereo import continuous_fully_connected


def convolution_3x1x1_with_relu(number_of_input_features,
                                number_of_output_features):
    return nn.Sequential(
        nn.Conv3d(number_of_input_features,
                  number_of_output_features,
                  kernel_size=(3, 1, 1),
                  padding=(1, 0, 0)),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))


class ContinuousFullyConnected(nn.Module):
    def __init__(self, number_of_features=64, number_of_hidden_layers=2):
        super(ContinuousFullyConnected, self).__init__()
        self._continuous_fully_connected = \
         continuous_fully_connected.ContinuousFullyConnected(
            number_of_features, number_of_hidden_layers)

    def forward(self, events_fifo):
        """Returns events projection.

        Args:
            events_fifo: first-in, first-out events queue of size
                        (batch_size, 2, number_of_events, height, width).

        Returns:
            projection: events projection of size (batch_size,
                        number_of_features, height, width).
        """
        # Note that polarity comes after timestamp in the fifo.
        (events_timestamps,
         events_polarity) = (events_fifo[:, 0, ...].unsqueeze(1),
                             events_fifo[:, 1, ...].unsqueeze(1))
        projection = self._continuous_fully_connected(events_timestamps,
                                                      events_polarity)
        return functional.relu(projection, inplace=True)


class TemporalConvolutional(nn.Module):
    def __init__(self, number_of_features=64):
        super(TemporalConvolutional, self).__init__()
        projection_modules = [
            convolution_3x1x1_with_relu(2, number_of_features)
        ]
        projection_modules += [
            convolution_3x1x1_with_relu(number_of_features, number_of_features)
            for _ in range(2)
        ]
        self._projection_modules = nn.ModuleList(projection_modules)

    def forward(self, events_fifo):
        """Returns events projection.

        Args:
            events_fifo: first-in, first-out events queue of size
                        (batch_size, 2, 7, height, width).

        Returns:
            projection: events projection of size (batch_size,
                        number_of_features, height, width).
        """
        projection = events_fifo
        for projection_module in self._projection_modules:
            projection = projection_module(projection)
        return projection.max(dim=2)[0]
