This repo contains the implementation of our paper:

## Context-aware Pedestrian Trajectory Prediction with Multimodal Transformer

Haleh Damirchi, Michael Greenspan, Ali Etemad

**ICIP 2023**  
[[paper](https:...)]

## Overview

![results](/image/method.jpg)

We propose a novel solution for predicting future trajecto
ries of pedestrians. Our method uses a multimodal encoder-
decoder transformer architecture, which takes as input both
pedestrian locations and ego-vehicle speeds. Notably, our de
coder predicts the entire future trajectory in a single-pass and
does not perform one-step-ahead prediction, which makes the
method effective for embedded edge deployment. We per
form detailed experiments and evaluate our method on two
popular datasets, PIE and JAAD. Quantitative results demon
strate the superiority of our proposed model over the current
state-of-the-art, which consistently achieves the lowest error
for 3 time horizons of 0.5, 1.0 and 1.5 seconds. Moreover,
the proposed method is significantly faster than the state-of-
the-art for the two datasets of PIE and JAAD. Lastly, ablation
experiments demonstrate the impact of the key multimodal
configuration of our method.

## Dataset
1. Please download the PIE dataset from the following link:  [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)

2. Modify the directories at configs/pie to reflect your directory containing the dataset.

## Installation

* **Tested OS**: Windows, Linux
* python 3.8
* pytorch 1.12

## Training

To train the model:

```
python train_deterministic.py --batch_size 128 --version_name Model --hidden_size_traj 256 --hidden_size_sp 128 --d_model_traj 256 --d_model_sp 128 --d_inner 1024 --d_k 32 --d_v 32 --n_head 16 --epochs 200 --patience 10
```

## Evaluation

To evaluate the model:

```
python eval_deterministic.py --batch_size 128 --version_name Model --hidden_size_traj 256 --hidden_size_sp 128 --d_model_traj 256 --d_model_sp 128 --d_inner 1024 --d_k 32 --d_v 32 --n_head 16 --epochs 200 --patience 10 --checkpoint checkpoint_address
```


References:


https://github.com/jadore801120/attention-is-all-you-need-pytorch

https://github.com/ChuhuaW/SGNet.pytorch

# Citation
If you find our work useful in your research, please cite our paper:
```bibtex
update
```