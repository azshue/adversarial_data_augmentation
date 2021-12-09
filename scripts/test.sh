#!/bin/bash

# the path to the directory where various datasets are stored
dataset_root=path/to/your/datasets
# directory of the dataset for training
dataset=name_of_the_dataset
# specify the name of the checkpoint directory
exp_name=train_adv_aug
# specify the checkpoint (epoch)
epoch=999

# test single factor degredations
python test_single.py --exp_name ${exp_name} --dataset_root ${dataset_root} --dataset ${dataset} --ckpt_epoch ${epoch};
# test combined factor degredations 
python test_comb.py --exp_name ${exp_name} --dataset_root ${dataset_root} --dataset ${dataset} --ckpt_epoch ${epoch}