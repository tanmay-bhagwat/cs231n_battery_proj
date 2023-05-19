#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os
from utils import *
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F

# cwd to directory containing the data as json files
os.chdir(r'/Users/tanmay/Desktop/CS231N/Project/Code_files/')

# Generating the dataset
# Select the test cell, pass to loader()
# loader(1)

# tr = ToTensor()
batt_dataset = BatteryDataset(csv_file='./image_dir/Training_data.csv', root_dir='./image_dir')

# To visualize a few samples from the dataset
dataset_visualize(batt_dataset)

# num_workers < number of cores on computer
# batch_size in multiples of 8, since internally PyTorch stores data as 8-bit integers
dataloader = DataLoader(batt_dataset, batch_size=32, shuffle=True, num_workers=4)
