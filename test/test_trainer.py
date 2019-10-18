# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile

from torch.optim import lr_scheduler
from torch.utils import data
import torch as th

from dense_deep_event_stereo import dataset
from dense_deep_event_stereo import trainer

from practical_deep_stereo import loss
from practical_deep_stereo import transformers as pds_transformers

from test import dataset_mockup


def _initialize_dataloaders(dataset_folder, temporal_aggregation_type,
                            queue_capacity, queue_time_horizon):
    mockup_dataset = dataset.IndoorFlying.dataset(
        dataset_folder,
        experiments=dataset_mockup.MOCKUP_MVSEC_STEREO_EXPERIMENT)
    mockup_dataset.subsample(1)
    dataset_transformers = trainer.initialize_transformers(
        temporal_aggregation_type, queue_capacity, use_full_ground_truth=False)
    # Crop a bit for speed up.
    if temporal_aggregation_type == 'hand_crafted':
        dataset_transformers += [
            pds_transformers.CentralCrop(
                height=64,
                width=64,
                get_items_to_crop=lambda x: [
                    x['left']['event_image'], x['left']['disparity_image'], x[
                        'right']['event_image']
                ])
        ]
    else:
        dataset_transformers += [
            pds_transformers.CentralCrop(
                height=64,
                width=64,
                get_items_to_crop=lambda x: [
                    x['left']['event_queue'], x['left']['disparity_image'], x[
                        'right']['event_queue']
                ])
        ]
    mockup_dataset.set_time_horizon(queue_time_horizon)
    mockup_dataset._transformers = dataset_transformers
    training_set_loader = data.DataLoader(mockup_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=3)
    validation_set_loader = data.DataLoader(mockup_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=3)
    return training_set_loader, validation_set_loader


def _initialize_parameters(dataset_folder, temporal_aggregation_type,
                           use_shallow_network):
    (main_lr, temporal_aggregation_lr, queue_time_horizon,
     queue_capacity) = trainer.default_network_dependent_parameters(
         temporal_aggregation_type, use_shallow_network)
    (training_set_loader, validation_set_loader) = _initialize_dataloaders(
        dataset_folder, temporal_aggregation_type, queue_capacity,
        queue_time_horizon)
    stereo_network = trainer.initialize_network(temporal_aggregation_type,
                                                use_shallow_network)
    optimizer = trainer.initialize_optimizer(stereo_network, main_lr,
                                             temporal_aggregation_lr)
    criterion = loss.SubpixelCrossEntropy()
    if th.cuda.is_available():
        criterion.cuda()
        stereo_network.cuda()
    return {
        'network':
        stereo_network,
        'optimizer':
        optimizer,
        'criterion':
        criterion,
        'learning_rate_scheduler':
        lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
        'training_set_loader':
        training_set_loader,
        'test_set_loader':
        validation_set_loader,
        'end_epoch':
        2,
        'experiment_folder':
        tempfile.mkdtemp()
    }


def test_trainer():
    dataset_folder = dataset_mockup.create_mvsec_dataset_mockup()
    for use_shallow_network in [True, False]:
        for temporal_aggregation_type in [
                'hand_crafted', 'temporal_convolutional',
                'continuous_fully_connected'
        ]:
            parameters = _initialize_parameters(dataset_folder,
                                                temporal_aggregation_type,
                                                use_shallow_network)
            if temporal_aggregation_type == 'hand_crafted':
                event_stereo_trainer = trainer.TrainerForHandcrafted(
                    parameters)
            else:
                event_stereo_trainer = trainer.Trainer(parameters)
            event_stereo_trainer.train()
            event_stereo_trainer.test()
