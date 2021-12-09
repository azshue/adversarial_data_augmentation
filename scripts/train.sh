#!/bin/bash

# the path to the directory where various datasets are stored
dataset_root=path/to/your/datasets
# directory of the dataset for training
dataset_name=name_of_the_dataset
# specify the name of the checkpoint directory
exp_name=train_adv_aug

# hyperparameters for data augmentation
# each number/letter represent an augmentation op (see data/diifaug.py)
augment_ops=N12RGBHSV
adv_stepsize=0.2
adv_eps=0.5
adv_n_step=3


python train.py --output_name ${exp_name} --dataset_root ${dataset_root} --dataset ${dataset_name} \
    --augments ${augment_ops} --adv_step ${adv_stepsize} --eps ${adv_eps} --n_repeats ${adv_n_step}