"""
This scripts uses a csv file with the names of the datasets we want the header information of.
The datasets can be either from openneuro or from the NeuroPoly lab.
The finality of the script is to create a csv file containing the header information of all the images in the datasets.
Note that the script has been designed to be able to be run several times in parralel.

/!\ The script uses the aws command line tool to download the headers of the images. Make you configured a aws account with the command "aws configure"
"""

from pathlib import Path
import subprocess
import os
import pandas as pd
import shutil
import nibabel as nib
import numpy as np
import random
import string



def fetch_contrast(filename_path):
    '''
    Extract MRI contrast from a BIDS-compatible IMAGE filename/filepath
    The function handles images only.
    :param filename_path: image file path or file name. (e.g sub-001_ses-01_T1w.nii.gz)
    Copied from https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    '''
    return filename_path.rstrip(''.join(Path(filename_path).suffixes)).split('_')[-1]

def orientation_string_nib2sct(s):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :return: SCT reference space code from nibabel one
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
    return "".join([opposite_character[x] for x in s])

def row_from_path(img_path, header_name):
    '''
    Extract all the required informations to file a row in the csv from the path of an image
    '''
    # Load the image
    img = nib.load(header_name+'.hdr')

    # Get the shape of the image
    shape = img.shape

    # Get the orientation of the image
    affine = img.affine
    res = "".join(nib.orientations.aff2axcodes(affine))
    orientation = orientation_string_nib2sct(res)

    # Get the resolution of the image
    p = img.header.get_zooms()
    is_2D = len(shape) == 2
    is_3D = len(shape) == 3
    is_4D = len(shape) == 4

    contrast = fetch_contrast(img_path)
    return img_path, contrast, orientation, shape, is_2D, is_3D, is_4D, p

def fetch_all_image_paths(dataset_path):
    '''
    Fetch all the image paths in the dataset
    '''
    images_paths=[]
    for p in Path(dataset_path).rglob('*.nii.gz'):
        if 'derivatives' not in str(p) and "code" not in str(p) and ".git" not in str(p):
            images_paths.append(str(p))
    return images_paths

def clone_dataset(dataset_name):
    '''
    Clone the dataset in the current directory
    '''
    subprocess.run(f"datalad clone https://github.com/OpenNeuroDatasets/{dataset_name}.git", shell=True)

def dl_header(img_path, file_name):
    '''
    Download the header of an image file in a openneuro dataset
    '''
    subprocess.run("aws s3api get-object --bucket openneuro.org --key "+img_path+" --range bytes=0-999 "+file_name+" && zcat "+file_name+" | head -c 348 > "+file_name+".hdr", shell=True)


#create a new dataframe
headers_df = pd.DataFrame(columns=['img_path', 'contrast', 'orientation', 'shape','is_2D', 'is_3D', 'is_4D','p'])

#load the csv containing already processed datasets
datasets_to_process = pd.read_csv('datasets_to_process.csv')['openneuro_dataset_id'].values

# if header_data directory does not exist, create it
if not os.path.exists('headers_data'):
    os.makedirs('headers_data')
    

# while the length of the processed datasets is less than the length of the dataset names
while len(datasets_to_process) > 0:
    # draw a dataset name randomly
    dataset_name = np.random.choice(datasets_to_process)

    # remove the dataset name from the list of datasets to process
    datasets_to_process = np.delete(datasets_to_process, np.where(datasets_to_process == dataset_name))

    #save the datasets to process
    pd.DataFrame(datasets_to_process, columns=['openneuro_dataset_id']).to_csv('datasets_to_process.csv', index=False)

    clone_dataset(dataset_name)
    print(f'Processing {dataset_name} dataset')
    img_paths = fetch_all_image_paths(dataset_name)
    for img_path in img_paths:
        # header name is a random sequence of characters
        header_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=15))
        #add a new row to the csv file
        dl_header(img_path, header_name)
        headers_df.loc[len(headers_df)] = row_from_path(img_path, header_name)

        # remove the header files
        subprocess.run(f'rm {header_name}', shell=True)
        subprocess.run(f'rm {header_name}.hdr', shell=True)

    #save the dataframe to a csv file
    print(f'Saving data until {dataset_name}')
    headers_df.to_csv(f'headers_data/headers_data_{dataset_name}.csv', index=False)
    


    #if the directory still exists
    if os.path.exists(dataset_name):
        #delete the dataset
        shutil.rmtree(dataset_name)

    # reload datasets to process
    datasets_to_process = pd.read_csv('datasets_to_process.csv')['openneuro_dataset_id'].values

# regroup all the csv files in one
all_headers = pd.read_csv('images_infos_until_biobank.csv')

# Reform the list of datasets to process
processed_datasets = pd.DataFrame(columns=['openneuro_dataset_id'])

for p in Path('headers_data').rglob('*.csv'):
    all_headers = pd.concat([all_headers, pd.read_csv(p)])
    processed_datasets.loc[len(processed_datasets)] = p.stem.split('_')[-1]

all_headers.to_csv('all_headers.csv', index=False)
processed_datasets.to_csv('processed_datasets.csv', index=False)






