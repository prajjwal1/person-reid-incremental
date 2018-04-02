# learning-without-forgetting-hybrid
#### Author: [Prajjwal](prajjwal1.github.io) (@prajjwal1)

## This is a documentation of this research
## Note
This repo contains the following files:
- Densely connected Convolutional Neural Networks (Implementation of this [paper](https://arxiv.org/abs/1608.06993))
- Fastai library and lwfnet (lib aimed to get SOTA results)
- Person Re-identification

## Pending
- [ ] Checking how well model performs. Plotting the accuracies, and doing statistical analysis.
- [ ] Working and finetuning the model if it helps.
- [ ] Working at backend to encapsulate the features of LWF
- [ ] Check how well model remembers
- [ ] Working on FC layers and learned weights to perform incremental learning.

## Plan (Research Workflow)
Firstly I am working on Person-Reidentiication (Since that is the primary goal) and later on I'll be working on  FC layers and a linear combination of previously learned weights when we add more layers. ie, increasing FC layers like LWF does and using a linear combination as one of the three papers do.
   - [Conditional Similarity Networks](https://arxiv.org/abs/1603.07810)
   - [Learning multiple visual domains with residual adapters](https://arxiv.org/abs/1705.08045)
   - [Incremental Classifier Learning with Generative Adversarial Networks (iCaRL) ](https://arxiv.org/abs/1802.00853)

  One other paper similar to Lwf : [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)

The work process would be initially work upon Person Re-identification model in an accurate manner and get reasonable results as per hardware availability and then research at back end to modify the model and inculcate the results of LWF with Incremental learning.

## Trained model
I've trained ResNet50 on Market1501. It's available to download [here](https://drive.google.com/open?id=1__x0qNJ3T654wTghmuRjydn42NsAZW_M)


## Requirements
- [Pytorch](pytorch.org) (>=0.3)
- Numpy
- Pandas
- Scikit-learn
- #### Nvidia GPU (Tesla K20 or better) (RAM >= 6GB)

### Downloading the Dataset
You can download the dataset through this [link](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing) (Google Drive).

### Managing the Dataset
After downlading the dataset, in the same directory extract the dataset and then Requirements
`
$ python dataset.py
`
### Training
This part is in progress, although you can run

`
$ python train.py
`

Training process has been haulted due to CUDA memory error,

## Dataset being used
- Market 1501

## License
This repository is private and will not be open sourced until research is completed
