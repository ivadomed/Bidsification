from torch.utils.data import DataLoader
from data_manager import Dataset_2D, dataset_splitter, paths_to_Dataset
from contrast_classifier_Networks import ResNet18SingleChannel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import argparse
import os


### Retrieve the arguments
# Create the parser
parser = argparse.ArgumentParser(description='Train and evaluate the model.')

# Add the arguments
parser.add_argument('--evaluate', type=bool, default=True,
                    help='a boolean for the evaluation mode, True by Default')
parser.add_argument('--data_csv', type=str, default='images_paths.csv',
                    help='the path to the dataset CSV file, images_paths.csv by Default')
parser.add_argument('--model_path', type=str, default='none',
                    help='the path to the model, Random weights by Default')
parser.add_argument('--output_file', type=str, default='model.pth',
                    help='the path to the output model, model.pth by Default')
parser.add_argument('--output_directory', type=str, default='checkpoints',
                    help='the path to the output directory, checkpoints by Default')
parser.add_argument('--base_dir', type=str, default='data',
                    help='the path to the base directory, data by Default')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='the number of epochs, 10 by Default')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
evaluate = args.evaluate
data_csv = args.data_csv
model_path = args.model_path
output_file = args.output_file
output_directory = args.output_directory + '/' 
base_dir = args.base_dir + '/'
num_epochs = args.num_epochs

# Define the training loop
def training_one_epoch(model):
    model.train()
    running_loss = 0.0
    queue_line = np.arange(train_dataset.length)
    np.random.shuffle(queue_line) 
    index=0
    for i in queue_line:
        image, label = train_dataset[i]
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        index += 1
    return model, running_loss / len(train_dataset)

#### Load the data

# Split the data into training and validation sets
pd_train_data, pd_val_data, num_classes = dataset_splitter(data_csv, train_ratio=0.8, random_seed=42)

# Create the training and validation datasets
train_dataset = paths_to_Dataset(pd_train_data, num_classes=num_classes, base_dir=base_dir)
val_dataset = paths_to_Dataset(pd_val_data, val=True, num_classes=num_classes, base_dir=base_dir)

#### Train the model

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18SingleChannel(num_classes=num_classes).to(device)
if model_path != 'none':
    model.load_state_dict(torch.load(model_path), map_location=device)
    print("Model loaded")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} / {num_epochs}")
    model, train_loss = training_one_epoch(model)
    print(f"Epoch {epoch + 1} training loss: {train_loss}")
    if (epoch + 1) % 10 == 0:
        # check if the directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        torch.save(model.state_dict(), output_directory + f"model_epoch_{epoch + 1}.pth")
#save model
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
torch.save(model.state_dict(), output_directory + output_file)

#### Evaluate the model

if evaluate:
    # Assess accuracy and F1 score on the validation set
    val_predictions = []
    val_labels = []
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        image, label = image.to(device), label.to(device)
        output = model(image)
        prediction = torch.round(output)
        val_predictions.append(prediction.item())
        val_labels.append(label.item())

    val_predictions = np.array(val_predictions)
    val_labels = np.array(val_labels)

    accuracy = np.mean(val_predictions == val_labels)
    f1_score = f1_score(val_labels, val_predictions)
