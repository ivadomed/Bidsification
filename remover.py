import os
import subprocess

def remove_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.chmod(os.path.join(root, file), 0o750)  # Give read, write, and execute permissions
        for dir in dirs:
            os.chmod(os.path.join(root, dir), 0o750)  # Give read, write, and execute permissions
    subprocess.run(f"rm -r {path}", shell=True)

remove_directory("/home/ge.polymtl.ca/p120530/Neuro_Poly_Training/3D_multi_contrast/Bidsification/data")
