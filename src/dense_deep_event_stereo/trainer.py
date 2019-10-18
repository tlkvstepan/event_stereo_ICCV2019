# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import torch as th
from torch import optim

from dense_deep_event_stereo import dataset_constants
from dense_deep_event_stereo import network
from dense_deep_event_stereo import transformers

from practical_deep_stereo import errors
from practical_deep_stereo import pds_trainer
from practical_deep_stereo import trainer
from practical_deep_stereo import visualization

NUMBER_OF_EVENTS_IN_SPARSE_GROUND_TRUTH = 15000


class Trainer(pds_trainer.PdsTrainer):
    def _initialize_filenames(self):
        super(Trainer, self)._initialize_filenames()
        self._left_events_template = os.path.join(
            self._experiment_folder, 'example_{0:02d}_events.png')
        self._nonmasked_estimated_disparity_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:02d}_disparity_nomask_epoch_{1:03d}.png')

    def _run_network(self, batch_or_example):
        batch_or_example['network_output'] = self._network(
            batch_or_example['left']['event_queue'],
            batch_or_example['right']['event_queue'])

    def _compute_error(self, example):
        estimated_disparity = example['network_output']
        ground_truth_disparity = example['left']['disparity_image']
        original_dataset = self._test_set_loader.dataset
        estimated_depth = original_dataset.disparity_to_depth(
            estimated_disparity)
        ground_truth_depth = original_dataset.disparity_to_depth(
            ground_truth_disparity)
        binary_error_map, one_pixel_error = errors.compute_n_pixels_error(
            estimated_disparity, ground_truth_disparity, n=1.0)
        mean_disparity_error = errors.compute_absolute_error(
            estimated_disparity, ground_truth_disparity)[1]
        mean_depth_error = errors.compute_absolute_error(
            estimated_depth, ground_truth_depth)[1]
        median_depth_error = errors.compute_absolute_error(estimated_depth,
                                                           ground_truth_depth,
                                                           use_mean=False)[1]
        example['binary_error_map'] = binary_error_map
        example['error'] = {
            'one_pixel_error': one_pixel_error,
            'mean_disparity_error': mean_disparity_error,
            'mean_depth_error': mean_depth_error,
            'median_depth_error': median_depth_error
        }

    def _report_training_progress(self):
        """Plot and print training loss and validation error every epoch."""
        test_errors = list(
            map(lambda element: element['one_pixel_error'], self._test_errors))
        visualization.plot_losses_and_errors(self._plot_filename,
                                             self._training_losses,
                                             test_errors)
        self._logger.log('epoch {0:02d} ({1:02d}) : '
                         'training loss = {2:.5f}, '
                         'mean disparity error = {3:.3f} [pix], '
                         '1PE = {4:.2f} [%], '
                         'mean depth error = {5:.3f} [m], '
                         'median depth error = {6:.3f} [m], '
                         'learning rate = {7:.5f}.'.format(
                             self._current_epoch + 1, self._end_epoch,
                             self._training_losses[-1],
                             self._test_errors[-1]['mean_disparity_error'],
                             self._test_errors[-1]['one_pixel_error'],
                             self._test_errors[-1]['mean_depth_error'],
                             self._test_errors[-1]['median_depth_error'],
                             trainer.get_learning_rate(self._optimizer)))

    def _report_test_results(self, error, time):
        self._logger.log('Testing results: '
                         'mean disparity error = {0:.3f} [pix], '
                         '1PE = {1:.3f} [%], '
                         'mean depth error = {2:.3f} [m], '
                         'median depth error = {3:.3f} [m], '
                         'time-per-image = {4:.2f} [sec].'.format(
                             error['mean_disparity_error'],
                             error['one_pixel_error'],
                             error['mean_depth_error'],
                             error['median_depth_error'], time))

    def _visualize_example(self, example, example_index):
        """Save visualization for examples.

        Visualizes estimated and ground truth disparity and
        event sequence for the left camera.
        """
        if example_index <= self._number_of_examples_to_visualize:
            # Get events sequence for left camera without transformers.
            left_events = self._test_set_loader.dataset.get_example(
                example_index)['left']['event_sequence']
            left_image = th.stack([example['left']['image'][0].data.cpu()] * 3,
                                  dim=0)
            left_events_visualization = th.from_numpy(
                left_events.to_image(background=left_image.numpy()))
            visualization.save_image(
                filename=self._left_events_template.format(example_index + 1),
                image=left_events_visualization,
                color_first=True)
            visualization.save_image(
                filename=self._left_image_template.format(example_index + 1),
                image=th.stack([example['left']['image'][0].cpu()] * 3, dim=2),
                color_first=False)
            # Dataset loader adds additional singletone dimension at the
            # beggining of tensors.
            ground_truth_disparity_image = example['left']['disparity_image'][
                0].cpu()
            estimated_disparity_image = example['network_output'][0].cpu()
            # Ensures same scale of the ground truth and estimated disparity.
            # Mask disparities not avaliable in ground truth.
            noninf_mask = ~th.isinf(ground_truth_disparity_image)
            minimum_disparity = ground_truth_disparity_image.min()
            maximum_disparity = ground_truth_disparity_image[noninf_mask].max()
            visualization.save_matrix(
                filename=self._ground_truth_disparity_image_template.format(
                    example_index + 1),
                matrix=ground_truth_disparity_image,
                minimum_value=minimum_disparity,
                maximum_value=maximum_disparity)
            visualization.save_matrix(
                filename=self._nonmasked_estimated_disparity_image_template.
                format(example_index + 1, self._current_epoch + 1),
                matrix=estimated_disparity_image,
                minimum_value=minimum_disparity,
                maximum_value=maximum_disparity)
            estimated_disparity_image[~noninf_mask] = float('inf')
            visualization.save_matrix(
                filename=self._estimated_disparity_image_template.format(
                    example_index + 1, self._current_epoch + 1),
                matrix=estimated_disparity_image,
                minimum_value=minimum_disparity,
                maximum_value=maximum_disparity)


class TrainerForHandcrafted(Trainer):
    def _run_network(self, batch_or_example):
        batch_or_example['network_output'] = self._network(
            batch_or_example['left']['event_image'],
            batch_or_example['right']['event_image'])


def initialize_optimizer(stereo_network, main_lr, temporal_aggregation_lr):
    if isinstance(stereo_network, network.ShallowEventStereo):
        return optim.RMSprop(
            [{
                "params": stereo_network._temporal_aggregation.parameters(),
                "lr": temporal_aggregation_lr
            }, {
                "params": stereo_network._spatial_aggregation.parameters()
            }, {
                "params": stereo_network._matching.parameters()
            }], main_lr)
    else:
        return optim.RMSprop(
            [{
                "params": stereo_network._temporal_aggregation.parameters(),
                "lr": temporal_aggregation_lr
            }, {
                "params": stereo_network._embedding.parameters()
            }, {
                "params": stereo_network._matching.parameters()
            }, {
                "params": stereo_network._regularization.parameters()
            }], main_lr)


def default_network_dependent_parameters(temporal_aggregation_type,
                                         use_shallow_network):
    (main_lr, temporal_aggregation_lr, queue_time_horizon,
     queue_capacity) = (None, None, None, None)
    if temporal_aggregation_type == 'hand_crafted':
        if use_shallow_network:
            (main_lr, temporal_aggregation_lr, queue_time_horizon,
             queue_capacity) = (1e-3, 1e-3, 0.1, float('inf'))
        else:
            (main_lr, temporal_aggregation_lr, queue_time_horizon,
             queue_capacity) = (1e-2, 1e-2, 0.1, float('inf'))
    elif temporal_aggregation_type == 'continuous_fully_connected':
        if use_shallow_network:
            # TODO: recheck
            (main_lr, temporal_aggregation_lr, queue_time_horizon,
             queue_capacity) = (1e-3, 1e-3, 0.5, 7)
        else:
            # TODO: recheck
            (main_lr, temporal_aggregation_lr, queue_time_horizon,
             queue_capacity) = (1e-4, 1e-3, 0.5, 1)
    elif temporal_aggregation_type == 'temporal_convolutional':
        if use_shallow_network:
            # TODO: recheck
            (main_lr, temporal_aggregation_lr, queue_time_horizon,
             queue_capacity) = (1e-3, 1e-3, 0.5, 7)
        else:
            # TODO: recheck
            (main_lr, temporal_aggregation_lr, queue_time_horizon,
             queue_capacity) = (1e-4, 1e-4, 0.5, 1)
    return (main_lr, temporal_aggregation_lr, queue_time_horizon,
            queue_capacity)


def initialize_network(temporal_aggregation_type, use_shallow_network):
    if use_shallow_network:
        network_class = network.ShallowEventStereo
    else:
        network_class = network.DenseDeepEventStereo
    if temporal_aggregation_type == 'continuous_fully_connected':
        return network_class.default_with_continuous_fully_connected()
    elif temporal_aggregation_type == 'temporal_convolutional':
        return network_class.default_with_temporal_convolutions()
    elif temporal_aggregation_type == 'hand_crafted':
        return network_class.default_with_hand_crafted()
    else:
        raise ValueError('wrong temporal temporal aggregation type')


def initialize_transformers(temporal_aggregation_type, queue_capacity,
                            use_full_ground_truth):
    single_view_transformers = [transformers.absolute_time_to_relative]
    # If "hand_crafted" temporal embedding, convert events sequence to
    # events image, overwise convert to events queue.
    if temporal_aggregation_type == 'hand_crafted':
        single_view_transformers += [transformers.EventSequenceToEventImage()]
    elif temporal_aggregation_type in [
            'continuous_fully_connected', 'temporal_convolutional'
    ]:
        single_view_transformers += [
            transformers.normalize_polarity,
            transformers.EventSequenceToEventQueue(
                queue_capacity=queue_capacity,
                image_height=dataset_constants.IMAGE_HEIGHT,
                image_width=dataset_constants.IMAGE_WIDTH),
        ]
    else:
        raise ValueError('wrong temporal aggregation type')
    dataset_transformers = []
    # If we train with last 15`000 events, mask ground truth for other events.
    if not use_full_ground_truth:
        dataset_transformers += [
            transformers.KeepRecentEventsGoundtruth(
                number_of_events=NUMBER_OF_EVENTS_IN_SPARSE_GROUND_TRUTH)
        ]
    dataset_transformers += [
        transformers.ApplyTransformersToLeftRight(single_view_transformers),
        transformers.dictionary_of_numpy_arrays_to_tensors
    ]
    return dataset_transformers
