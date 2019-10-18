#!/usr/bin/env python3.6
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main script for training and testing a network.

    Learning rate, queue capacity and time horizon are automatically set based
    on the type of the temporal embedding.

    Args:
        experiment_folder: output folder, where training checkopoints,
                           log and plot will be saved.
        dataset_folder: folder with converted MVSEC dataset. By default it is
                        set to "dataset/".
        checkpoint_file: it this option is set, the script will load network
                         from the checkpoint. This option can be use if we
                         want to start training from the checkopoint or
                         run test.
        temporal_aggregation_type: type of temporal aggregation. Can be equal
                                   to "hand_crafted", "temporal_convolutional"
                                   or "continuous_fully_connected". By default
                                   it is "continuous_fully_connected".
        test_mode: if this flag is set, the script runs test set.
        use_full_ground_truth: if flag is set, a full ground truth is used for
                               the training. By default, only ground truth of
                               the 150`000 proceeding events is used for
                               the training.
        split_number: number of split, used during training. Can be equal to 1
                      or 3. By default it is equal to 1.
        queue_capacity: number of events stored in each location. If this
                        parameter is not provided, it is set to default based
                        on "temporal_aggregation_type" and
                        "use_shallow_network" parameters.
        queue_time_horizon: lifetime of an oldest event in the queue. If this
                      parameter is not provided, it is set to default based on
                      "temporal_aggregation_type" and "use_shallow_network"
                      parameters.
        use_shallow_network: If this flag is set, then instead of PDS network
                             with a large receptive field we use MC-CNN network
                             with a small 9 x 9 receptive field.
        main_lr: learning rate of the network. If this parameter is not
                 provided it is set to default based on
                 "temporal_aggregation_type" and "use_shallow_network"
                 parameters.
        temporal_aggregation_lr: learning rate of the temporal aggregation
                                 module. If this parameter is not provided it
                                 is set to default based on
                                 "temporal_aggregation_type" and
                                 "use_shallow_network" parameters.
        debug_mode: if this flag is set, traininig is performed for 2 epoches.
                    This can be usefull for parameter selection.

    Example call:

    ./run_experiment.py "experiments/continuous_fully_connected" \
        --dataset_folder "dataset"
        --temporal_aggregation_type continuous_fully_connected
        --split_number 1
"""
import click
import os

from torch.optim import lr_scheduler
from torch.utils import data
import torch as th

from dense_deep_event_stereo import dataset
from dense_deep_event_stereo import trainer

from practical_deep_stereo import loss


def _initialize_dataloaders(dataset_folder, split_number, queue_time_horizon,
                            temporal_aggregation_type, queue_capacity,
                            use_full_ground_truth, test_mode, debug_mode):
    sets = dataset.IndoorFlying.split(dataset_folder,
                                      split_number=split_number)
    training_set = sets[0]
    if test_mode:
        test_set = sets[2]
    else:
        test_set = sets[1]
    dataset_transformers = trainer.initialize_transformers(
        temporal_aggregation_type, queue_capacity, use_full_ground_truth)
    training_set.set_time_horizon(queue_time_horizon)
    test_set.set_time_horizon(queue_time_horizon)
    training_set._transformers = dataset_transformers
    test_set._transformers = dataset_transformers
    training_set_loader = data.DataLoader(training_set,
                                          batch_size=1,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=3)
    test_set_loader = data.DataLoader(test_set,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=3)
    return training_set_loader, test_set_loader


def _initialize_parameters(dataset_folder, temporal_aggregation_type,
                           experiment_folder, split_number, queue_capacity,
                           queue_time_horizon, use_full_ground_truth,
                           test_mode, use_shallow_network, main_lr,
                           temporal_aggregation_lr, debug_mode):
    training_set_loader, test_set_loader = _initialize_dataloaders(
        dataset_folder, split_number, queue_time_horizon,
        temporal_aggregation_type, queue_capacity, use_full_ground_truth,
        test_mode, debug_mode)
    stereo_network = trainer.initialize_network(temporal_aggregation_type,
                                                use_shallow_network)
    optimizer = trainer.initialize_optimizer(stereo_network, main_lr,
                                             temporal_aggregation_lr)
    learning_rate_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 10, 12, 14, 16, 18, 20, 22], gamma=0.5)
    if use_shallow_network:
        criterion = loss.SubpixelCrossEntropy(diversity=1.0, disparity_step=1)
    else:
        criterion = loss.SubpixelCrossEntropy(diversity=1.0, disparity_step=2)
    if th.cuda.is_available():
        criterion.cuda()
        stereo_network.cuda()
    if debug_mode:
        end_epoch = 2
    else:
        end_epoch = 22
    return {
        'network': stereo_network,
        'optimizer': optimizer,
        'criterion': criterion,
        'learning_rate_scheduler': learning_rate_scheduler,
        'training_set_loader': training_set_loader,
        'test_set_loader': test_set_loader,
        'end_epoch': end_epoch,
        'experiment_folder': experiment_folder
    }


@click.command()
@click.argument('experiment_folder', type=click.Path(exists=False))
@click.option('--checkpoint_file', default=None, type=click.Path(exists=True))
@click.option('--dataset_folder',
              default='dataset',
              type=click.Path(exists=True))
@click.option('--temporal_aggregation_type',
              type=click.Choice([
                  'hand_crafted', 'temporal_convolutional',
                  'continuous_fully_connected'
              ]),
              default='continuous_fully_connected')
@click.option('--use_full_ground_truth', is_flag=True)
@click.option('--test_mode', is_flag=True)
@click.option('--split_number', type=click.Choice(['1', '3']), default='1')
@click.option('--queue_time_horizon', type=click.FLOAT, default=None)
@click.option('--queue_capacity', type=click.INT, default=None)
@click.option('--use_shallow_network', is_flag=True)
@click.option('--main_lr', type=click.FLOAT, default=None)
@click.option('--temporal_aggregation_lr', type=click.FLOAT, default=None)
@click.option('--debug_mode', is_flag=True)
def main(experiment_folder, checkpoint_file, dataset_folder,
         temporal_aggregation_type, use_full_ground_truth, test_mode,
         split_number, queue_time_horizon, queue_capacity, use_shallow_network,
         main_lr, temporal_aggregation_lr, debug_mode):
    (default_main_lr, default_temporal_aggregation_lr,
     default_queue_time_horizon,
     default_queue_capacity) = trainer.default_network_dependent_parameters(
         temporal_aggregation_type, use_shallow_network)
    if main_lr is None:
        main_lr = default_main_lr
    if temporal_aggregation_lr is None:
        temporal_aggregation_lr = default_temporal_aggregation_lr
    if queue_time_horizon is None:
        queue_time_horizon = default_queue_time_horizon
    if queue_capacity is None:
        queue_capacity = default_queue_capacity
    dataset_folder = os.path.abspath(dataset_folder)
    experiment_folder = os.path.abspath(experiment_folder)
    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)
    parameters = _initialize_parameters(
        dataset_folder, temporal_aggregation_type, experiment_folder,
        int(split_number), queue_capacity, queue_time_horizon,
        use_full_ground_truth, test_mode, use_shallow_network, main_lr,
        temporal_aggregation_lr, debug_mode)
    if temporal_aggregation_type == 'hand_crafted':
        stereo_trainer = trainer.TrainerForHandcrafted(parameters)
    else:
        stereo_trainer = trainer.Trainer(parameters)
    if checkpoint_file:
        stereo_trainer.load_checkpoint(checkpoint_file, load_only_network=True)
    if test_mode:
        stereo_trainer.test()
    else:
        stereo_trainer.train()


if __name__ == '__main__':
    main()
