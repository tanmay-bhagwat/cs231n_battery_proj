#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from utils import *
from BatteryDataset import *
from Net_model import *
import torch.optim as optim
from torch.utils.data import DataLoader, sampler, Subset

# cwd to directory containing the data as json files
os.chdir(r'/Users/tanmay/Desktop/CS231N/Project/Code_files/')

# Generating the dataset
# Select the test cell, mode, pass to build_dataset()
mode = "C1"
test_cell = 1 
build_dataset(test_cell, path=".", slice_len=900, mode=mode)

batt_dataset = BatteryDataset(csv_file=f'./image_dir_{mode}/Training_data.csv', root_dir=f'./image_dir_{mode}')
test_dataset = BatteryDataset(csv_file=f'./image_dir_{mode}/Test_data.csv', root_dir=f'./image_dir_{mode}')

# To visualize a few samples from the dataset
# dataset_visualize(batt_dataset, mode="OCV")

# num_workers < number of cores on computer
# batch_size in multiples of 8, since internally PyTorch stores data as 8-bit integers
# Paper divided dataset in a 70:30 split, batch_size=128, for epochs=80
NUM_TRAIN = int(len(batt_dataset)*0.7)

train_loader = DataLoader(batt_dataset, batch_size=128, sampler = sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                        num_workers=4)
validation_loader = DataLoader(batt_dataset, batch_size=128, sampler = sampler.SubsetRandomSampler(range(NUM_TRAIN, len(batt_dataset))),
                        num_workers=4)
test_loader = DataLoader(batt_dataset, batch_size=128, num_workers=4)

# Enumerate and print the batches that each of the loaders has
# if __name__ == '__main__':
#     for i_batch, sample_batched in enumerate(train_loader):
#         print(i_batch, sample_batched[0].size(), sample_batched[1].size())
#         if sample_batched[0].dtype != torch.float32:
#             raise ValueError('Incorrect dtypes')
#         # print(type(sample_batched[0]))

#     # Define a model and the optimizer
#     model = Net()
#     # L2 regularization via weight_decay: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
#     optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
#     # train_model(model, optimizer, train_loader, val_loader=validation_loader, epochs=20)
#     # torch.save(model.state_dict(), 'Net30.pth')

#     loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
#     model = Net()
#     model.load_state_dict(torch.load('Net30.pth'))
#     test_model(model, loader)