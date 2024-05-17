import pandas as pd
import numpy as np
import subprocess
import os
import shutil
from pathlib import Path
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the arguments
parser.add_argument('--max_image_per_contrast', type=int, default=1000,
                    help='The maximum number of images per contrast')
parser.add_argument('--min_image_per_contrast', type=int, default=100,
                    help='The minimum number of images per contrast')
parser.add_argument('--evaluate', type=bool, default=True,
                    help='Whether to evaluate or not')

# Parse the arguments
args = parser.parse_args()

# Get the arguments
max_image_per_contrast = args.max_image_per_contrast
min_image_per_contrast = args.min_image_per_contrast
evaluate = args.evaluate

def get_dataset(path):
    if '\s' in path:
        return(path.split('\s')[0])
    else:
        return(path.split('/')[0])

# Load the header data csv
header_data = pd.read_csv('all_headers.csv')

header_data['dataset'] = header_data['img_path'].apply(get_dataset)

# only keep 3D image files
header_data = header_data[header_data['is_3D']]

# Create an empty df to store the selected headers
selected_headers = pd.DataFrame(columns=['img_path', 'contrast', 'orientation', 'shape','is_2D', 'is_3D', 'is_4D','p', 'dataset'])

# List all remaining contrasts*
contrasts = header_data['contrast'].unique()

# Only keep Constrast with more than the given min
contrasts = [c for c in contrasts if len(header_data[header_data['contrast'] == c]) > min_image_per_contrast]

# For all contrasts with 
for contrast in contrasts:
    contrast_count = len(header_data[header_data['contrast'] == contrast])

    # if less than the given max,
    if contrast_count <= max_image_per_contrast:
        # keep all images
        selected_headers = pd.concat([selected_headers, header_data[header_data['contrast'] == contrast]])

    # if more than the given max,
    if contrast_count > max_image_per_contrast:
        selected_headers_contrast = pd.DataFrame(columns=['img_path', 'contrast', 'orientation', 'shape','is_2D', 'is_3D', 'is_4D','p', 'dataset'])

        header_data_contrast=header_data[header_data['contrast']==contrast]

        # list all datasets with this contrast
        datasets = header_data_contrast['dataset'].unique()

        # Count the number of images with the contrast "contrast" in each dataset
        
        dataset_counts = header_data_contrast['dataset'].value_counts()
        # sort dataset_counts by the number of images
        dataset_counts = dataset_counts.sort_values(ascending=True)

        sample_per_dataset = max_image_per_contrast // len(datasets)

        # for each dataset, in the ascending order of the number of images
        for i, dataset in enumerate(dataset_counts.index):
            # if the count is less than the sample_per_dataset
            if dataset_counts[dataset] < sample_per_dataset:
                # keep all images
                selected_headers_contrast = pd.concat([selected_headers_contrast, header_data_contrast[header_data['dataset'] == dataset]])
                # update the sample_per_dataset
                sample_per_dataset = (max_image_per_contrast - len(selected_headers_contrast)) // (len(datasets) - i)
            # if the count is more than the sample_per_dataset
            else :
                # keep sample_per_dataset images
                selectin = header_data_contrast[header_data_contrast['dataset'] == dataset].sample(sample_per_dataset)
                selected_headers_contrast = pd.concat([selected_headers_contrast, header_data_contrast[header_data_contrast['dataset'] == dataset].sample(sample_per_dataset)])

        selected_headers = pd.concat([selected_headers, selected_headers_contrast])
# count the number of images per contrast
contrast_counts = selected_headers['contrast'].value_counts()

# associate a probability to each image so the dataset is balanced
selected_headers['p_draw'] = selected_headers['contrast'].apply(lambda x: 1/(contrast_counts[x] * len(selected_headers)))

# Save the selected headers
selected_headers.to_csv('selected_headers.csv', index=False)


if evaluate:
    # in order to test the absence of correlation between the features of the images and the contrast, we wilt train a random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # remove img_path, is_2D, is_3D, is_4D, dataset, orientation columns
    X = selected_headers.drop(columns=['img_path', 'is_2D', 'is_3D', 'is_4D', 'dataset', "contrast", "orientation"])
    Y = selected_headers['contrast']
    sample_weights = selected_headers['p_draw']

    # in X transform shape (x,y,z) to three columns as well as p (px, py, pz)
    X['shape_x'] = X['shape'].apply(lambda x: float(x.split(',')[0][1:]))
    X['shape_y'] = X['shape'].apply(lambda x: float(x.split(',')[1]))
    X['shape_z'] = X['shape'].apply(lambda x: float(x.split(',')[2][0:-1]))
    X['p_x'] = X['p'].apply(lambda x: float(x.split(',')[0][1:]))
    X['p_y'] = X['p'].apply(lambda x: float(x.split(',')[1]))
    X['p_z'] = X['p'].apply(lambda x: float(x.split(',')[2][0:-1]))
    X = X.drop(columns=['shape', 'p'])
    
    # normalize the data, column by column
    for column in X.columns:
        X[column] = (X[column] - X[column].mean()) / X[column].std()


    # split the data
    X_train, X_test, Y_train, Y_test, sw_train, sw_test = train_test_split(X, Y, sample_weights, test_size=0.2)

    # create the model
    clf = RandomForestClassifier()

    # train the model with sample weights
    print("Training the model...")
    clf.fit(X_train, Y_train, sample_weight=sw_train)

    # evaluate the model
    print("Evaluating the model...")
    Y_pred = clf.predict(X_test)
    print(Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy : ", accuracy)
    
    # in order to compare, we run a random classifier as well
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy='uniform')
    clf.fit(X_train, Y_train, sample_weight=sw_train)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy of the random classifier : ", accuracy)




