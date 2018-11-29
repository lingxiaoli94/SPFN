## Supervised Fitting of Geometric Primitives to 3D Point Clouds
Lingxiao Li*, [Minhyuk Sung](http://mhsung.github.io)*, [Anastasia Dubrovina](http://web.stanford.edu/~adkarni/), [Li Yi](https://cs.stanford.edu/~ericyi/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)
(* indicates equal contribution)
[[arXiv](https://arxiv.org/abs/1811.08988)]

### Introduction
Fitting geometric primitives to 3D point cloud data bridges a gap between low-level digitized 3D data and high-level structural information on the underlying 3D shapes. As such, it enables many downstream applications in 3D data processing. For a long time, RANSAC-based methods have been the gold standard for such primitive fitting problems, but they require careful per-input parameter tuning and thus do not scale well for large datasets with diverse shapes. In this work, we introduce Supervised Primitive Fitting Network (SPFN), an end-to-end neural network that can robustly detect a varying number of primitives at different scales without any user control. The network is supervised using ground truth primitive surfaces and primitive membership for the input points. Instead of directly predicting the primitives, our architecture first predicts per-point properties and then uses a differentiable model estimation module to compute the primitive type and parameters. We evaluate our approach on a novel benchmark of ANSI 3D mechanical component models and demonstrate a significant improvement over both the state-of-the-art RANSAC-based methods and the direct neural prediction.

### Requirements
The code has been tested with Tensorflow 1.10 (GPU version) and Python 3.6.6. There are a few dependencies on the following Python libraries : numpy (tested with 1.14.5), scipy (tested with 1.1.0), pandas (tested with 0.23.4), PyYAML (tested with 3.13), and h5py (tested with 2.8.0).

### Usage

#### Dataset
First, while staying in the project folder, download processed ANSI 3D dataset of mechanical components (9.4GB zip file, 12GB after unzipping):
```
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/minhyuk/spfn/data/spfn_traceparts.zip
unzip spfn_traceparts.zip
```
The original CAD data is kindly provided by [Traceparts](https://www.traceparts.com). The provided dataset has been processed to extract primitive surface informations and samples on each surface as well as on the whole shape.

#### Training
Train SPFN with our default configuration by:
```
mkdir experiments && cd experiments
python3 ../spfn/train.py ../default_configs/network_config.yml
```
Note that the script `train.py` takes a configuration YAML file `network_config.yml` that contains GPU setting, data source, neural network parameters, training hyperparameters, and I/O parameters. Simply copy the default YAML configuration file and change parameters to your need. During training, three folders will be created/updated. In their default locations, `results/model` is the directory for storing the Tensorflow model, `results/log` is the directory for log files (created by `tf.summary.FileWriter`), and `results/val_pred` contains predictions for the validation dataset at varying training steps.

#### Test
At test time, run `train.py` with `--test` flag to run the network on test dataset speficied by `test_data_file` in the YAML configuration:
```
python3 ../spfn/train.py ../default_configs/network_config.yml --test 
```
As a shortcut, and also to test the network with only input points without other supervision, run `train.py` with `--test_pc_in=tmp.xyz` and `--test_h5_out=tmp.h5` in additional to `--test` flag, where `tmp.xyz` is assumed to be a point cloud file with one point `x y z` on each line:
```
python3 ../spfn/train.py ../default_configs/network_config.yml --test --test_pc_in=tmp.xyz --test_h5_out=tmp.h5
```
The predictions are stored in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format. Each HDF5 prediction file contains per-point normal prediction, per-point membership prediction, per-point type prediction, and estimated parameters for each primitive.

#### Evaluation
