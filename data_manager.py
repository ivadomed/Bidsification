import numpy as np
import nibabel as nib
import pandas as pd
import os
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
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset_from_path(path):
    if '\s' in path:
        return(path.split('\s')[0])
    else:
        return(path.split('/')[0])

def clone_dataset(dataset_name, is_openneuro=False):
    '''
    Clone the dataset in the current directory
    '''       

    if is_openneuro:
        stream = os.popen(f"datalad clone https://github.com/OpenNeuroDatasets/{dataset_name}.git")
        stream = os.popen(f"datalad siblings -d {os.getcwd()}/{dataset_name} enable -s s3-PRIVATE")
    else:
        stream = os.popen(f"git clone git@data.neuro.polymtl.ca:datasets/{dataset_name}.git")

def download_sample(img_path):
    '''
    Download the sample from the dataset with git annex
    '''
    # change directory to the correct dataset folder
    os.chdir(dataset_from_path(img_path))
    # remove dataset name from the image path
    img_path = img_path.split(dataset_from_path(img_path)+"/")[1]
    stream = os.popen(f"datalad get {img_path}")

    # change directory back to the "data" folder
    os.chdir("..")

def dl_dataset(dataset_path_csv):
    # Load the dataset from the CSV file
    dataset_paths = pd.read_csv(dataset_path_csv)

    # add a column to the dataset with the name of the dataset of origin
    dataset_paths['dataset'] = dataset_paths['img_path'].apply(dataset_from_path)
    
    #make a list of the datasets
    datasets = dataset_paths['dataset'].unique()

    # clone each dataset repository
    if not os.path.exists("data"):
        os.makedirs("data")
    #change the current directory to the "data" folder
    os.chdir("data")
    for dataset in datasets:
        clone_dataset(dataset, is_openneuro=dataset.startswith('ds0'))
        i+=1
    print("Cloning done")
    
    for img_path in dataset_paths['img_path']:
        download_sample(img_path)
    


# Define a custom dataset class
class Dataset_2D(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.data = {"paths" : paths, "labels" : labels}
        self.transform = transform
        self.length = len(self.data["paths"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data["paths"][index]
        label = [0,1] if self.data["labels"][index] else [1,0]
        if self.transform:
            image = self.transform(path)
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

        # convert label list to tensor with shape [1,2]
        label = torch.tensor([label])
        return image, label

def find_T1w_T2w_paths(base_dir):
    """ Find the relative paths of the T1w and T2w files in the specified directory."""
    #base_dir="data//data-multi-subject//"

    desired_extension = ".json"

    # Initialize lists to store the relative paths for T1w, T2w, and DWI files
    t1w_file_paths = []
    t2w_file_paths = []

    print("Searching for T1w, T2w, and DWI files in", base_dir, "...")

    # Traverse the directory structure
    for root, dirs, files in os.walk(base_dir):
        # Exclude the "derivatives" subfolder
        if "derivatives" in dirs:
            dirs.remove("derivatives")
        for file in files:
            # Check if the file name contains the desired names
            if "T1w" in file and file.endswith(desired_extension):
                # Get the relative path of the T1w file
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                # Remove the file extension
                relative_path = os.path.splitext(base_dir + relative_path)[0] + ".nii.gz"
                # Append the relative path to the T1w file paths list
                t1w_file_paths.append(relative_path)
            elif "T2w" in file and file.endswith(desired_extension):
                # Get the relative path of the T2w file
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                # Remove the file extension
                relative_path = os.path.splitext(relative_path)[0] + ".nii.gz"
                # Append the relative path to the T2w file paths list
                t2w_file_paths.append(base_dir + relative_path)

    #t1w_file_paths = t1w_file_paths[:20]
    #t2w_file_paths = t2w_file_paths[:20]

    print("Found", len(t1w_file_paths), "T1w files and", len(t2w_file_paths), "T2w files.")

    return t1w_file_paths, t2w_file_paths


def dataset_splitter(t1w_file_paths, t2w_file_paths, train_ratio=0.8, random_seed=42):
    """ Split the dataset into training and validation sets based on the specified ratio and random seed."""
    path_data = pd.DataFrame({"image_path" : t1w_file_paths + t2w_file_paths, "labels" : len(t1w_file_paths) * [0] + len(t2w_file_paths) * [1]})

    pd_train_data, pd_val_data = train_test_split(path_data, test_size=0.2, random_state=random_seed)
    pd_train_data.reset_index(drop=True, inplace=True)
    pd_val_data.reset_index(drop=True, inplace=True)

    return pd_train_data, pd_val_data


def paths_to_Dataset(pd_data, val = False):
    """ Convert the file paths to a custom dataset object."""
    if val:
        transform = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        RandShiftIntensity(offsets=0.1, prob=0.5),
        RandRotate(range_x=3, range_y=3, range_z=3, prob=0.2),
        
    ]
)
    else:
        transform = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
    ]
)
    dataset = Dataset_2D(pd_data["image_path"], pd_data["labels"], transform=transform)
    return dataset