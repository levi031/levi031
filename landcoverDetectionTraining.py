import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, randome_split
import matplotlib.pyplot as plt
from datetime import datetime

from ConvolutionNeuralNetwork import ResUNET, ImprovedUNet
from Datasets import MultiModalSegmentationDataset

#Contents
#1.0 Configuration



#1.0 Configuration
#1.1 Config Block
#Easy set up for model runs from the model type, aspp module, DEM, learning rate
#Epochs, validations, etc. Can cahnge these around to test out optimal model configuration.
#
config = {
    'model_type': 'resunet', #select between A. 'resunet' or B. 'improved_unet'
    'use_aspp': True, # ASPP Module is for multi-scale processing
    'include_dem': True, #Turn on and off for testing
    'include_topography_calculations': True, #slope, aspect, Topography Position Index
    'batch_size': 4,
    'epochs': 25, #test model at low epoch, final runs at high
    'learning_rate': 1e-4, 
    'lambda_4class': 1.0, #weight for 4-class track (RESEARCH)
    'lambda_binary': 0.3, #weight for the binary (RESEARCH)
    'val_fraction': 0.2, #train fraction will just be 1-val_fraction
    'log_interval': 1, #Printing after this many epochs
}

#1.2 Study area and date
year = "2020"
prov = 'Tshopo'

#1.3 Project path
#Training script location
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

#1.4 Results directory
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

try:
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results Directory: {results_dir}")
except PermissionError:
    print(f"Permission denied in {results_dir}")
    print("Using alternative")
    results_dir = r"C:\Users\levim\thesisModel"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Using: {results_dir}")
except Exception as exception:
    print(f"Error creating directory: {exception}")
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#1.5 Data directories
HLS_dir = os.path.join(project_root, "data", "satellite", str({year}), str({prov}))
train_dir = os.path.join(project_root, "data", "train", str({year}), str({prov}))
DEM_dir = os.path.join(project_root, "data", "DEM", str({year}), str({prov}))

print("Model start")
print("Model features")
print(f"Start {config['model_type']}")
print(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"DEM included: {config['include_dem']}")
print(f"Derived features: {config['include_topography_calculations']}")
print(f"ASPP module: {config['use_aspp']}")
print("...")

