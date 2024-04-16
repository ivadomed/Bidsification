# Bidsification

This repo aims at creating a pipeline which could determine de bidnomenclature of a MRI image file from it's content. 

The first part consist in training the required contrast classifier models.
The goal of this part if to build robust networks that can take 2D 1 channel MRI image of any shape, any view point, any field of view and output it's contrast.

The first network, in its dedicated branch, is designed to discriminate T1w against T2w.

## First T1W / T2w classifier (proof of concept)

### Usage

In order to launch the training, one can use this command :
`python train.py --evaluate True --model_path /path/to/model.pth --model_output path/to/model_out.pth`

In order to launch the testing, one can use this command :
`python train.py --evaluate True --model_path /path/to/model.pth --model_output path/to/model_out.pth`

### Dataset

The original training set is extracted from the public dataset [Spine Generic](https://github.com/spine-generic/data-multi-subject).
This Dataset contains 3D images and require proper prepocessing to be used.
The spine generic dataset is meant to be stored like data/data-multi-subject/..

### Preprocessing

The data has been preprocesse to make the network as robust as possible

* The dataset is filtered to keep T1w and T2w images paths only.
* The dataset is splited between train patients and test patients (20% test)
* Two object from the class "2D_dataset" are created. They encapsulate the label and the image path.
* At each training epoch, the model sees each 3D image once. Each time the image is randomly :
    - flipped
    - rotated (in a 3° range)
    - Shifted (in a 0.1 range)
    - reframed (in a 2D fashion, with minimum size (30 * 30))

### Network Structure

The network used is a ResNet18 from pytorch library modified to handle 1 channel images and to output a 2D vector.