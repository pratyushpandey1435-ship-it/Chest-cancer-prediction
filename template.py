import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

project_name = "cnnClassifier" # can give any model name

# Project Structure
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py", # name of the folders to be created with their constructors
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]
# For Handling if any path related error 
#  it will change the path according to compatability to the operating system
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "": # if folder exist then make file in it 
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f: # creating file with suitable extension given 
            pass
            logging.info(f"Creating empty file: {filepath}")

    else: # if file already exist
        logging.info(f"{filename} is already exist")