# Bidsification

This repo aims at creating a pipeline which could determine de bidnomenclature of a MRI image file from it's content. 

The first part consist in training the required contrast classifier models.
The goal of this part if to build robust networks that can take 2D 1 channel MRI image of any shape, any view point, any field of view and output it's contrast.

The first network, in its dedicated branch, is designed to discriminate T1w against T2w.

## First T1W / T2w classifier (proof of concept)

### Dataset

This model is meant to be trained with a dataset selected by the scripts of the branch "Dataset_selection". It provides a "selected_header.csv" file with relevant files names to be found in several dataset from NeuroPoly and [OpenNeuro](https://openneuro.org/). 

### Usage

One first needs to download the dataset described by "selected_header.csv" :
`python download_dataset.py --dataset_scv_file selected_header.csv`

In order to launch the training, one can use this command :
`python train.py --evaluate True --dataset_csv_file selected_headers.csv --model_path /path/to/model.pth --model_output path/to/model_out.pth`

### Preprocessing

The data has been preprocesse to make the network as robust as possible

* The dataset is splited between train patients and test patients (20% test)
* Two object from the class "2D_dataset" are created. They encapsulate the label and the image path.
* At each training epoch, the model sees each 3D image once. Each time the image is randomly :
    - flipped
    - rotated (in a 15° range)
    - Shifted (in a 0.1 range)
    - reframed (in a 2D fashion, with minimum size (30 * 30))

### Network Structure

The network used is a ResNet18 from pytorch library modified to handle 1 channel images and to output a 2D vector.