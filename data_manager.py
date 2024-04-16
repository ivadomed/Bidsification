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

    pd_train_data, pd_val_data = train_test_split(path_data, test_size=0.2, random_state=0)
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