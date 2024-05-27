# This script is designed to force remove a directory and all its contents. 
# It is handy to use if teh downloded dataset refuses to  be removed because of git annex restrictions.
# /!\ Be careful, this script will remove the directory and all its contents permanently.


import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Remove the data directory.')
parser.add_argument('--directory_to_remove', type=str, default=None,
                    help='the path to the data directory, None by Default')

args = parser.parse_args()
directory_to_remove = args.directory_to_remove


def remove_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.chmod(os.path.join(root, file), 0o600)  # Give read and write permissions to the current user only
        for dir in dirs:
            os.chmod(os.path.join(root, dir), 0o600)  # Give read and write permissions to the current user only
    subprocess.run(f"rm -r {path}", shell=True)

remove_directory(directory_to_remove)
