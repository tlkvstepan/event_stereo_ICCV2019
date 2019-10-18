# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
cimport numpy as np


cpdef compute_spatial_binning(np.ndarray[double, ndim=3] sequences,
                   np.ndarray[long, ndim=1] sequences_lengths,
                   np.ndarray[long, ndim=1] features_indices,
                   int height, int width, int maximum_local_sequence_length,
                   int y_column, int x_column):
    """Returns spatially binned events.

    If number of events in location exceeds "number_of_events_per_location"
    additonal events are not considered.

    Args:
        sequences: 3d numpy array with indices
                  [sequence_index, event_index, feature_index].
                  Inside the sequences, the event are arranged from
                  oldest to newest.
        sequences_lengths 1D length of the sequence.
        height, width: dimensions of the underlaying spatial grid.
        maximum_local_sequence_length: maximum number of events per spatial
                      location. If number of events exceeds this value the
                      oldest event in the location is deleted to and all events
                      are shifted to the begginning to free some space.
        x_column, y_column: indices of features containing x and y position of
                            the event.
    Returns:
        local_sequences: 5d numpy array with indices [sequence_index, y, x,
                         event_index, feature_index]. It stores
                         events sequences in every location (x, y).
        local_sequences_lengths: 3d numpy array with indices
                        [sequence_index, y, x]. It stores number of events
                        in every location.
    """
    cdef int event_index, local_event_index, feature_index, sequence_index
    cdef int feature_index_in_local_sequence, feature_index_in_sequence
    cdef int sequence_length
    cdef number_of_features = features_indices.shape[0]
    cdef number_of_sequences = sequences.shape[0]
    cdef int x, y
    cdef np.ndarray[double, ndim=5] local_sequences
    cdef np.ndarray[long, ndim=3] local_sequences_lengths
    local_sequences = np.zeros((number_of_sequences, height, width,  maximum_local_sequence_length,
                                number_of_features), dtype=np.float)
    local_sequences_lengths = np.zeros((number_of_sequences, height, width), dtype=np.int)
    for sequence_index in range(number_of_sequences):
        sequence_length = int(sequences_lengths[sequence_index])
        for event_index in range(sequence_length):
            x = int(sequences[sequence_index, event_index, x_column])
            y = int(sequences[sequence_index, event_index, y_column])
            number_of_events_in_location = int(local_sequences_lengths[sequence_index, y, x])
            if number_of_events_in_location >= maximum_local_sequence_length:
               continue
            local_event_index = int(local_sequences_lengths[sequence_index, y, x])
            for feature_index_in_local_sequence in range(number_of_features):
                feature_index_in_sequence = features_indices[feature_index_in_local_sequence]
                local_sequences[sequence_index, y, x, local_event_index, feature_index_in_local_sequence] = \
                    sequences[sequence_index, event_index, feature_index_in_sequence]
            local_sequences_lengths[sequence_index, y, x] += 1
    return local_sequences, local_sequences_lengths


cpdef compute_spatial_hash_table(np.ndarray[double, ndim=2] events_sequence,
                                         int y_index, int x_index,
                                         int image_height, int image_width):
    """Returns spatial hash for events sequence.

    Args: 
        events sequence: 2D array, where each row corresponds to the event and
                         each column to the event's feature. The events are sorted
                         in oldest-first order.
    Returns:
        spatial_hash_table: 3D array, where [y, x, n] element contains index of the 
                      n-th event that arrive in (y, x).
        events_in_location: 2D array, where [y, x] element contains number of
                            events that arrive in (y, x) location.x
    """
    cdef int events_number = events_sequence.shape[0]
    cdef int features_number = events_sequence.shape[1]
    cdef int maximum_events_in_location
    maximum_events_in_location = compute_maximum_events_in_location(
        events_sequence, y_index, x_index, image_height, image_width)
    cdef int event_index
    cdef np.ndarray[long, ndim=3] spatial_hash_table = np.zeros(
        (image_height, image_width, maximum_events_in_location), dtype=np.int)
    cdef np.ndarray[long, ndim=2] events_in_location = np.zeros(
        (image_height, image_width), dtype=np.int)
    cdef int y, x
    for event_index in range(events_number):
        x = int(events_sequence[event_index, x_index])
        y = int(events_sequence[event_index, y_index])
        spatial_hash_table[y, x, events_in_location[y, x]] = event_index
        events_in_location[y, x] += 1

    return (spatial_hash_table, events_in_location)


cpdef int compute_maximum_events_in_location(
                    np.ndarray[double, ndim=2] events_sequence,
                    int y_column, int x_column,
                    int image_height, int image_width):
    """Computes maximum number of events in location.

    Args:
        events_sequence:  2D array where each row
                  correspond to event and columns to the events' features.
                  The events are sorted from oldest to newest.
        x_column, y_column: columns of the "events_sequence" table with x
                            and y coordinates of the event.
        image_width, image_height: sizes of the image.
    """
    cdef np.ndarray[long, ndim=2] events_in_location
    events_in_location = np.zeros((image_height, image_width), dtype=np.int)
    cdef int x, y
    cdef int events_number = events_sequence.shape[0]
    cdef int maximum_events_in_location = 0
    for event_index in range(events_number):
        x = int(events_sequence[event_index, x_column])
        y = int(events_sequence[event_index, y_column])
        events_in_location[y, x] += 1
    for x in range(image_width):
        for y in range(image_height):
            if maximum_events_in_location < events_in_location[y, x]:
                maximum_events_in_location = events_in_location[y, x]
    return maximum_events_in_location
