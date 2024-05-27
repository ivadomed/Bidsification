import numpy as np
import nibabel as nib
import pandas as pd
import os
import subprocess
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandRotate,
    RandShiftIntensity,
    ToTensor,
    RandSpatialCrop,
    LoadImage,
    SqueezeDim,
    RandRotate,
    RandSimulateLowResolution,
    Resize,
)
import torch
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset_from_path(path):
    if '\s' in path:
        print("el famoso")
        print(path.split('\s'))
        return(path.split('\s')[0])
    else:
        return(path.split('/')[0])

def clone_dataset(dataset_name, base_dir = "data/", is_openneuro=False):
    '''
    Clone the dataset in the current directory
    '''
    print(f"Cloning the dataset {dataset_name}...")       

    if is_openneuro:
        subprocess.run(f"git clone https://github.com/OpenNeuroDatasets/{dataset_name}.git", shell=True, cwd=base_dir)
        #subprocess.run(f"git-annex siblings -d {os.getcwd()}/{dataset_name} enable -s s3-PRIVATE", shell=True, cwd=base_dir)
    else:
        subprocess.run(f"git clone git@data.neuro.polymtl.ca:datasets/{dataset_name}.git", shell=True, cwd=base_dir)
        
def download_sample(img_path, dataset, base_dir = "data/"):
    '''
    Download the sample from the dataset with git annex
    '''
    # remove dataset name from the image path
    folder = os.path.dirname(base_dir + img_path)
    img_path = img_path.split(dataset_from_path(img_path)+"/")[1]
    print(f"Downloading {img_path}...")
    print(f"directory : {dataset}")
    if dataset.startswith('ds0'):
        # aws s3 cp s3://openneuro.org/ds004146/sub-0385/ses-02/anat/sub-0385_ses-02_T2w_TSE_run-02.nii.gz data/ds004146/sub-0385/ses-02/anat/sub-0385_ses-02_T2w_TSE_run-02.nii.gz
        subprocess.run(f"aws s3 cp s3://openneuro.org/{dataset}/{img_path} {folder}", shell=True)
    else :
        subprocess.run(f"git-annex get {img_path}", shell=True, cwd = base_dir + dataset)

def contrast_name_to_label(data_csv):
    contrast_names = data_csv['contrast'].unique()
    contrast_labels = {contrast_name : i for i, contrast_name in enumerate(contrast_names)}
    # add labels to the csv file
    data_csv['label'] = data_csv['contrast'].apply(lambda x : contrast_labels[x])
    return data_csv


def dl_dataset(dataset_path_csv, base_dir = "data/"):
    # Load the dataset from the CSV file
    dataset_paths = pd.read_csv(dataset_path_csv)

    # add a column to the dataset with the name of the dataset of origin
    dataset_paths['dataset'] = dataset_paths['img_path'].apply(dataset_from_path)
    
    #make a list of the datasets
    datasets = dataset_paths['dataset'].unique()

    # clone each dataset repository
    if not os.path.exists("data"):
        subprocess.run("mkdir data", shell=True)
    #change the current directory to the "data" folder
    for dataset in datasets:
        clone_dataset(dataset, base_dir = base_dir, is_openneuro=dataset.startswith('ds0'))
    print("Cloning done")
    
    for row in dataset_paths.iterrows():
        img_path, dataset = row[1]['img_path'], row[1]['dataset']
        download_sample(img_path, dataset, base_dir = base_dir,)
        print(f"Downloaded {img_path}")

    


# Define a custom dataset class
class Dataset_2D(Dataset):
    def __init__(self, paths, labels, transform=None, num_classes=2, base_dir="data/"):
        self.data = {"paths" : paths, "labels" : labels}
        self.transform = transform
        self.length = len(self.data["paths"])
        self.num_classes = num_classes
        self.base_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.base_dir + self.data["paths"][index]
        label = np.zeros(self.num_classes)
        label[self.data["labels"][index]] = 1

        if self.transform:
            ## Rotate, flip, shift intensity if training
            image = self.transform(path)

            ## Simulate low or high resolution
            input_shape = image.shape
            # draw a random shape between 0.5 * input_shape and 2 * input_shape
            target_shape = [int(np.random.uniform(0.5, 2) * s) for s in input_shape]

            # if one of the target shape dimension is smaller than the input shape dimension, we need to anti-aliasing
            anti_aliasing = False
            for i in range(3):
                if target_shape[i] < input_shape[i] :
                    anti_aliasing = True
                    break
            
            image=Resize(spatial_size=target_shape, mode="trilinear", anti_aliasing=anti_aliasing)(image)

            ## Random crop and squeeze to 2D
            dim_to_squeeze = int(np.random.choice([0,1,2]))
            roi_min = np.array([30, 30, 30])
            roi_max = np.array([-1, -1, -1])
            roi_min[dim_to_squeeze] = 1
            roi_max[dim_to_squeeze] = 1
            image = RandSpatialCrop(roi_min,  max_roi_size = roi_max, random_size=True, random_center=True)(image)
            image = SqueezeDim(dim=dim_to_squeeze + 1)(image)
            # Convert to tensor
            image = ToTensor()(image)
            # add a dimension to the image, for exemple [1, 256, 256] -> [1, 1, 256, 256]
            image = image.unsqueeze(0)

        # convert label list to tensor with shape [1,num_classes]
        label = torch.tensor([label])
        return image, label


def dataset_splitter(data_csv, train_ratio=0.8, random_seed=42):
    """ Split the dataset into training and validation sets based on the specified ratio and random seed."""
    pd_data = pd.read_csv(data_csv)
    pd_data = contrast_name_to_label(pd_data)
    pd_train_data, pd_val_data = train_test_split(pd_data, test_size=0.2, random_state=random_seed)

    pd_train_data.reset_index(drop=True, inplace=True)
    pd_val_data.reset_index(drop=True, inplace=True)
    # Get the number of classes
    num_classes = pd_data['label'].nunique()

    # normalize "p_draw" column so that the sum is 1 in train dataset
    pd_train_data['p_draw'] = pd_train_data['p_draw']/pd_train_data['p_draw'].sum()

    # set draw probability to 1/nb_classes in validation dataset
    pd_val_data['p_draw'] = 1/num_classes
    
    return pd_train_data, pd_val_data, num_classes


def paths_to_Dataset(pd_data, val = False, num_classes=2, base_dir="data/"):
    """ Convert the file paths to a custom dataset object."""
    if val:
        transform = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        RandShiftIntensity(offsets=0.1, prob=0.5),
        RandRotate(range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),      
    ]
)
    else:
        transform = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
    ]
)
    dataset = Dataset_2D(pd_data["img_path"], pd_data["label"], transform=transform, num_classes=num_classes, base_dir=base_dir)
    return dataset