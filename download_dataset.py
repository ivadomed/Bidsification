from data_manager import dataset_from_path, clone_dataset, download_sample, dl_dataset
import argparse

### Retrieve the arguments
# Create the parser
parser = argparse.ArgumentParser(description='Download the dataset discribed by the csv file.')

# Add the arguments
parser.add_argument('--dataset_csv_file', type=str, default='images_paths.csv',
                    help='the path to the dataset CSV file, images_paths.csv by Default')
parser.add_argument('--base_dir', type=str, default='data/',
                    help='the path to the base directory, data/ by Default')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_csv_file = args.dataset_csv_file
base_dir = args.base_dir

dl_dataset(dataset_csv_file, base_dir=base_dir)