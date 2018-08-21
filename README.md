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
Duke MTMC
```
$ python covariance_duke.py
```

This repo is under construction, will be documented gradually

### Citation:
Please cite this, if you use our work
```
@misc{1808.06281,
Author = {Prajjwal Bhargava},
Title = {Incremental Learning in Person Re-Identification},
Year = {2018},
Eprint = {arXiv:1808.06281},
}
```
