## Supervised Fitting of Geometric Primitives to 3D Point Clouds
Lingxiao Li*, [Minhyuk Sung](http://mhsung.github.io)*, [Anastasia Dubrovina](http://web.stanford.edu/~adkarni/), [Li Yi](https://cs.stanford.edu/~ericyi/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)

[[arXiv](https://arxiv.org/abs/1811.08988)]

### Abstract
Fitting geometric primitives to 3D point cloud data bridges a gap between low-level digitized 3D data and high-level structural information on the underlying 3D shapes. As such, it enables many downstream applications in 3D data processing. For a long time, RANSAC-based methods have been the gold standard for such primitive fitting problems, but they require careful per-input parameter tuning and thus do not scale well for large datasets with diverse shapes. In this work, we introduce Supervised Primitive Fitting Network (SPFN), an end-to-end neural network that can robustly detect a varying number of primitives at different scales without any user control. The network is supervised using ground truth primitive surfaces and primitive membership for the input points. Instead of directly predicting the primitives, our architecture first predicts per-point properties and then uses a differentiable model estimation module to compute the primitive type and parameters. We evaluate our approach on a novel benchmark of ANSI 3D mechanical component models and demonstrate a significant improvement over both the state-of-the-art RANSAC-based methods and the direct neural prediction.

### Requirements
The code has been tested with Tensorflow 1.10 (GPU version) and Python 3.6. There are a few dependencies on the following Python libraries : numpy (tested with 1.14.5), scipy (tested with 1.1.0), pandas (tested with 0.23.4), PyYAML (tested with 3.13), and h5py (tested with 2.8.0).

### Usage
First, while staying in the project folder, download processed ANSI 3D dataset of mechanical components:
```
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/minhyuk/spfn/data/spfn_traceparts.zip
unzip spfn_traceparts.zip
```
The original CAD data is kindly provided by [Traceparts](https://www.traceparts.com). The provided dataset has been processed to extract primitive surface informations and samples on each surface as well as on the whole shape.

Train SPFN with our default configuration by:
```
mkdir experiments && cd experiments
python3 ../spfn/train.py ../default_configs/network_config.yml
```
Note that the script `train.py` takes a configuration YAML file `network_config.yml` that contains GPU setting, data source, neural model parameters, training hyperparameters, and I/O parameters. Simply copy the default YAML configuration file and change parameters to your need.
