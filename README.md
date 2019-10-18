# Learning an event sequence embedding for dense event-based deep stereo (work in progress)

## Description
Asynchronous event sequences require special handling, since traditional algorithms work only with synchronous, spatially gridded data. To address this problem we introduce a new module for event sequence embedding, for use in different applications. The module builds a representation of an event sequence by firstly aggregating information locally across time, using a novel fully-connected layer for an irregularly sampled continuous domain, and then across discrete spatial domain. Based on this module, we design a deep learning-based stereo method for event-based cameras. The proposed method is the first learning-based stereo method for an event-based camera and the only method that produces dense results. We show large performance increases on the Multi Vehicle Stereo Event Camera Dataset (MVSEC). 

Please cite our [ICCV2019 paper](https://www.idiap.ch/~fleuret/papers/tulyakov-et-al-iccv2019.pdf) if you use this repository as 
```
@inproceedings{tulyakov-et-al-2019,
  author = {Tulyakov, S. and Fleuret, F. and Kiefel, M. and Gehler, P. and Hirsch, M.},
  title = {Learning an event sequence embedding for event-based deep stereo},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2019},
  type = {Oral},
  note = {To appear},
  url = {https://fleuret.org/papers/tulyakov-et-al-iccv2019.pdf}
}
```

## Prepare dataset

Please install [conda](https://www.anaconda.com/download).

Then, create new conda environment with python2.7 and all dependencies by running
```
conda config --append channels conda-forge
conda create --name  convert_mvsec --file tools/requirements.txt --yes python=2.7
```
Next, install  ``cv_bridge``, ``rosbag`` and  ``rospy`` packages from [Robot Operating System(ROS)](http://wiki.ros.org/indigo/Installation/Ubuntu), by following instructions on the web site.

Next, clone third party dependency ["The Multi Vehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception  by Zhu A. Z., Thakur D., Ozaslan T., Pfrommer B., Kumar V. and Daniilidis K."](https://daniilidis-group.github.io/mvsec/)  into `third_party` folder by running
```
git clone https://github.com/daniilidis-group/mvsec third_party/mvsec
```
and create software links in ``tools`` folder for all python scripts in `third_party/mvsec/gt_flow`. This is required since one of module   conflicts with one of the standard ROS modules and the conflict can not be resolved without modification of the third party code. 

Finally, download and convert ``indoor_flying 1,2,3,4`` experiments  by running:
```
conda activate convert_mvsec
tools/convert_mvsec.py "dataset" --experiment indoor_flying 1
tools/convert_mvsec.py "dataset" --experiment indoor_flying 2
tools/convert_mvsec.py "dataset" --experiment indoor_flying 3
```

## Install

Please install [conda](https://www.anaconda.com/download) 

Next, create new conda environment with python3.6 and all dependencies, including [pytorch](https://pytorch.org/), by running.
```
conda config --append channels pytorch
conda create --name dense_deep_event_stereo --file requirements.txt --yes python=3.6
```

Then,  install the package by running 
```
conda activate dense_deep_event_stereo
python setup.py develop
```

Next, install third party dependency - ["Practical Deep Stereo (PDS): Toward application-friendly deep stereo matching" by Tulyakov S., Ivanov A. and Fleuret F.](https://github.com/tlkvstepan/PracticalDeepStereo_NIPS2018.git) by running:

```
conda activate dense_deep_event_stereo
git clone https://github.com/tlkvstepan/PracticalDeepStereo_NIPS2018.git third_party/PracticalDeepStereo_NIPS2018
cd third_party/PracticalDeepStereo_NIPS2018
python setup.py develop
```

## Run experiments
All networks from the paper can be trained and tested using ``run_experiment.py`` script. 

For example, to train the network with a continuous fully connected temporal aggregation on split \#1, please, run
```
conda activate dense_deep_event_stereo
./run_experiment.py experiments/train_continuous_fully_connected \
 --dataset_folder dataset \ 
 --temporal_aggregation_type continuous_fully_connected \
 --split_number 1  
```
The results of the training with training log, plot and results for a validation set then can be found in `experiments/train_continuous_fully_connected` folder.

Next, to test the network from checkpoint `experiments/train_continuous_fully_connected/009_checkpoint.bin`, please run
``` 
conda activate dense_deep_event_stereo
./run_experiment.py experiments/test_continuous_fully_connected \
--dataset_folder dataset \
--checkpoint_file experiments/train_continuous_fully_connected/009_checkpoint.bin \
--temporal_aggregation_type continuous_fully_connected \
--split_number 1 \
--test_mode
```

## Troubleshooting

Please run all unit tests to localilze potential bugs by executing 
```
./run_tests.sh
```
