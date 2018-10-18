# Incremental Learning in Person Re-Identification

===========================================================

This repository contains code for our research. Paper can be found here, [arXiv](https://arxiv.org/abs/1808.06281)

## Getting started
1. `cd ~/PATH_NAME`
2. Run `git clone https://github.com/prajjwal1/person-reid-incremental`
3. Install the specified dependencies, to install use
` pip3 install - requirements.txt`
4. Follow the below mentioned steps for preparation of dataset and performing training

## Prerequisites:
-  OS: Linux/MacOS
-  Pytorch>=0.3

## Install Dependencies
- [Installing Pytorch](pytorch.org)

### Datasets
- [Market 1501](http://www.liangzheng.org/Project/project_reid.html)
- [Duke MTMC](http://vision.cs.duke.edu/DukeMTMC/)

### Dataset structure
This is the recommended file structure which was used

For preparation of Market1501
```
+-- Market1501
|   +-- bounding_box_test
|   +-- bounding_box_train
.......
```

For preparation of Duke MTMC
```
+-- dukemtmc-reid
  |   +-- DukeMTMC-reID
    |   +-- bounding_box_test
    |   +-- bounding_box_train
.............
```

Covariance loss metric has been added to all the modules. 
You're required to change the flags as per phase as described in paper

Create a directory named as data/ and use the standard directory structure.

For training on Market1501:
```
$ python covariance_market1501.py
```

For training on Duke MTMC. 
```
$ python covariance_duke.py
```

To use ensembling and training, use
```
$ python covariance_ensembling.py
```

In this case, you'll have to specify amongst which pipelines do you want to perform ensembling. If you get better results, please file a PR.

### To perform training:
While executing make sure to correctly carry out training (Phase 1 and Phase 2) properly as mentioned to achieve incremental learning

When training, log file would be created in the /log directory.

Results:

| No.|      Dataset      |  Rank 1 | Rank 20 | maP |
|---:|:-------------: |--------:|---------|-----|
| 1       | Market1501      | 89.3%  |  98.3%  |71.8%|
| 2       | DukeMTMC      | 80.0%  |  93.7%  |60.2%|
| 3       | Market1501      | 70.2%  |  92.4%  |41.2%|


Takes around 8-9 hours to train the model for 950 epochs (convergence is usually achieved)


### Models
We used a ResNet50 along with different architecture of pipelines. We have used `hybrid_convnet2`. You are required to change the dimensions of the FC layer as per number of classes manually. 

### To resume training
```
$ mkdir saved_models
```
Then specify this as per dir structure in the main module
```
SAVED_MODEL_PATH = 'saved_models/p1.pth.tar'
checkpoint = torch.load(SAVED_MODEL_PATH)
model.load_state_dict(checkpoint['state_dict'])
```
### For evaluation
```
$ python evaluation.py
```
Make sure to set the dataset and path of the models correctly, and also which pipeline to use for evaluation

### Citation:
Please cite this, if you use our work
```
@misc{bhargava2018incremental,
    title={Incremental Learning in Person Re-Identification},
    author={Prajjwal Bhargava},
    year={2018},
    eprint={1808.06281},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
