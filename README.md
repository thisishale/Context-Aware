This repo contains the implementation of our paper:

#Context-aware Pedestrian Trajectory Prediction with Multimodal Transformer"

Haleh Damirchi, Michael Greenspan, Ali Etemad

**ICIP 2023**  
[[paper](https:...)]

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