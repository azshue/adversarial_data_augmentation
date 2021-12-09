# Adversarial Differentiable Data Augmentation
This repository provides the official PyTorch implementation of the ICRA 2021 paper:     
> [Adversarial Differentiable Data Augmentation for Autonomous Systems](https://ieeexplore.ieee.org/document/9561205)      
> Author: *Manli Shu, Yu Shen, Ming C Lin, Tom Goldstein*      

## Environment
The code has been tested on:    
* python == 3.7.9
* pytorch == 1.10.0
* torchvision == 0.8.2
* kornia == 0.6.2  
More dependencies can be found at `./requirements.txt` 

Hardware requirements:     
* The default training and testing setting requires 1 GPU. 

## Data
Datasets appeared in our paper can be downloaded/generated by following the directions in [this page](https://github.com/YuShen0118/Multi_Perturbation_Robustness#dataset).    

*Note:* The "distortion" factor is added differently in our work, for which we cropped out the zero-padding around the distorted images. To reproduce the results in our paper, the same post-processing should be applied to the generated images with the "distortion" corruption:
```
python utils/cropping.py --dataset_root ${dataset_root} --dataset ${valData}
```
, where testing data with different corruptions are sorted in different folders under `${dataset_root}` and `${valData}` is the folder name of the original validation set without any corruption. 

## Training
1. Set the `${dataset_root}` and the `${dataset_name}` arguments in `./scripts/train.sh`. The "train" and "val" splits of the `${dataset_name}` are supposed to be stored separatly under `${dataset_root}`.
2. Set the hyper-parameters for data augmentation in `./scripts/train.sh`.
3. Run:
    ```
    bash ./scripts/train.sh
    ```

## Testing
1. Set the paths to your dataset in `./scripts/test.sh`
2. `exp_name`: help locating the model checkpoint (should be one of the training exp).
3. `epoch`: specify the model checkpoint
4. Run:      
    ```
    bash ./scripts/test.sh
    ``` 
Note that in the test script, we test the "combined" corrupting factor seperately, where we test a total of 25 random combination of corruptions. Test images with combined corrupting factors are generated on the fly, and we fix the random seed for reproducibility. (The randomly generated combination can be found in `./data/comb_param.txt`. )

## Citation
If you find the code or our method useful, please consider citing: 
```
@InProceedings{shu2021advaug,
    author={Shu, Manli and Shen, Yu and Lin, Ming C. and Goldstein, Tom},
    title={Adversarial Differentiable Data Augmentation for Autonomous Systems}, 
    booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
    year={2021}
}
```