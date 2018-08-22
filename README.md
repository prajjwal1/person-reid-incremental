# Incremental Learning in Person Re-Identification

This repository contains code for our research. Paper can be found here, [arXiv](https://arxiv.org/abs/1808.06281)

### Requirements:
- Pytorch>=0.3

### Datasets
- [Market 1501](http://www.liangzheng.org/Project/project_reid.html)
- [CUHK-03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- [Duke MTMC](http://vision.cs.duke.edu/DukeMTMC/)

Covariance loss metric has been added to all the modules. 
You're required to change the flags as per phase as described in paper

Create a directory named as data/ and use the standard directory structure.

For training on Market1501:
```
$ python covariance_market1501.py
```
CUHK-03
```
$python covariance_cuhk-03.py
```
Make sure that you are using the required split and flags, since training is CUHK-03 is more different than these two datasets
Duke MTMC. Although you can add support for arg parse
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
| 3       | Market1501      | 69.5%  |  92.8%  |40.3%|

| No.|      Dataset      |  Rank 1 | Rank 20 | maP |
|---:|:-------------: |--------:|---------|-----|
| 1       | Market1501      | 89.3%  |  98.3%  |71.8%|
| 2       | CUHK-03      | -  | - |-|
| 3       | Market1501      | - |  - |-|

Takes around 8-9 hours to train the model for 950 epochs (convergence is usually achieved)

More benchmarks would be released soon.

### Models
We used a ResNet50 along with different architecture of pipelines. We have used `hybrid_convnet2`.

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
### TO-DO
[ ] Add support for Tensorboard (Pytorch)
[ ] Usage of Random Erasing

### Citation:
Please cite this, if you use our work
```
@ARTICLE{2018arXiv180806281B,
   author = {{Bhargava}, P.},
    title = "{Incremental Learning in Person Re-Identification}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1808.06281},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition},
     year = 2018,
    month = aug,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180806281B},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
