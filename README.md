# Permutation-Aware Activity Segmentation via Unsupervised Frame-to-Segment
Alignment

For our recent works, please check out our research page (https://retrocausal.ai/research/).

## Overview

This repository implements UFSA, an unsupervised permutation-aware method for temporal action segmentation.
Details regarding the required environment, datasets, training scripts and pretrained models can be found below.
 <br />
  <br />
   <br />


## Enviroment
Pytorch == `1.10.0+cu102`, 
torchvision == `0.11.1`, 
python == `3.9.7`, 
CUDA==`10.2`

### Enviroment Setup
Install the required libaries as follows:

``` python
conda clean -a -y
conda create -n ufsa python=3.9.7 numpy
conda activate ufsa
conda install  --insecure pytorch=1.10.0 torchvision=0.11.1 torchaudio=0.10.0 cudatoolkit=11.3.1  -c pytorch
python -c "import torch; print(torch.__version__)"
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c conda-forge einops
conda install -c conda-forge torchinfo
conda install -c anaconda pandas
conda install -c conda-forge tensorboardx
conda install -c anaconda ipykernel
conda install ipython
conda install pip
```



#### Folders
For each dataset create separate folder (specify path --data_root) where the inner folders structure is as following:

> features/  
> groundTruth/  
> mapping/  
> models/

## Datasets

#### 50 Salads
- 50Salads features [link](https://drive.google.com/open?id=17o0WfF970cVnazrRuOWE92-OiYHEXTT3)
- 50Salads ground truth [link](https://drive.google.com/open?id=1mzcN9pz1tKygklQOiWI7iEvcJ1vJfU3R)

#### YouTube Instructions

- YouTube Instructions features [link](https://drive.google.com/open?id=1HyF3_bwWgz1QNgzLvN4J66TJVsQTYFTa) 
- YouTube Instructions ground truth [link](https://drive.google.com/open?id=1ENgdHvwHj2vFwflVXosCkCVP9mfLL5lP)

#### Breakfast

- Breakfast features [link](https://drive.google.com/file/d/1DbYnU2GBb68CxEt2I50QZm17KGYKNR1L)
- Breakfast ground truth [link](https://drive.google.com/file/d/1RO8lrvLy4bVaxZ7C62R0jVQtclXibLXU)

#### Desktop Assembly 

- Desktop Assembly features [link](https://drive.google.com/drive/folders/1t-dUAcY4QMbGt6xHEGriOMgSl5TRBXFM?usp=drive_link)
- Desktop Assembly ground truth [link](https://drive.google.com/drive/folders/1Ql3PwcR24hgjxzCX4XGvcQfVlhekqZu1?usp=drive_link)


## Training
We train the model in a two stages process:
In the first stage, we train the encoder using fixed order temporal optimal transport.
In the second stage, we train encoder+ transcript decoder along with cross-attention loss for alignment. 

All training scrips for all three datasets are provided with the [pretrained_models](pretrained_models). For each of the scripts you need to specify the `--data_root`. 
For more information regarding the flags, please look into  the information for each flag in `run.py`.
An example training script for `50salads` dataset is provided below:

<strong>Training script for stage 1:</strong>
``` python
python run.py --use_cuda  --dataset 50salads   --do_framewise_loss_gauss  
```

<strong>Training script for stage 2:</strong>
``` python
python run.py --use_cuda --dataset 50salads  --use_pe_tgt  --do_framewise_loss_gauss --do_segwise_loss --do_crossattention_action_loss_nll --pretrained_model /path to model/
```
### YTI

<strong>Training script for stage 1:</strong>
``` python
python run_yti.py --use_cuda  --dataset 50salads   --do_framewise_loss_gauss  
```

<strong>Training script for stage 2:</strong>
``` python
python run_yti.py --use_cuda --dataset 50salads  --use_pe_tgt  --do_framewise_loss_gauss --do_segwise_loss --do_crossattention_action_loss_nll 
```
Specify path to stage1 model for each action inside the loop via the `--pretrained_model` flag.


## Evaluation

To run the inference code the flag `--inference_only` needs to be added as well as `--path_inference_model` to point to the model to be evaluated.

An example script for testing a model is provided below:

<strong>Evaluate Stage 2:</strong>
``` python
python run.py --use_cuda --dataset 50salads --path_inference_model /path to model  --inference_only  --use_pe_tgt --use_transcript_dec 
```            

  


##  License

This project is open-sourced under the AGPL-3.0 license. See the [License](LICENSE) file for details.

For a list of other open source components included in this project, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the Project
This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.


