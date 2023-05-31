import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
import os
from Data_Preprocessor import *
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F


def build_dataset(test_cell, path, slice_len, mode="C1") -> None:
    """
    Function to call Data_Preprocessor object. Uses DataFrames to write the csv annotations files.

    Parameters:
    test_cell (int): The cell which is the test data
    path (str): The path to the directory which contains the json files
    slice_len (int): The slice length to be used for each cycle; must be perfect square
    mode (str): To denote if "C1" or "OCV" dataset to be used
    """
    # Selected cell (test_cell) as the test cell
    # df is then the training dataset DataFrame, can write it to a csv file and use in PyTorch Dataset class
    index_list = [i for i in range(1,9) if i != test_cell]

    if os.path.exists(os.path.join(path,f"./image_dir_{mode}")) != True:
        os.mkdir(f'./image_dir_{mode}')
    
    obj = Data_Preprocessor(path, slice_len, mode)
    dataset_dict = obj.data_processor()

    # Constructing the training and test annotation files
    df = pd.DataFrame()
    for k in index_list:
        df = pd.concat((df,pd.DataFrame.from_dict(dataset_dict[k])))

    # Normalizing data
    df['labels'] = (df['labels'] - df['labels'].mean())/(df['labels'].max() - df['labels'].min())
    
    df_test = pd.DataFrame.from_dict(dataset_dict[test_cell])
    # Normalizing data
    df_test['labels'] = (df_test['labels'] - df_test['labels'].mean())/(df_test['labels'].max() - df_test['labels'].min())

    df.to_csv(f'./image_dir_{mode}/Training_data.csv')
    df_test.to_csv(f'./image_dir_{mode}/Test_data.csv')


def dataset_visualize(batt_dataset) -> None:
    """Display images from the dataset."""

    for i in range(len(batt_dataset)):
        sample = batt_dataset[i]
        label = sample[1]

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i})')
        ax.set_xlabel(f'{label:0.5f}')
        plt.imshow(np.array(T.ToPILImage()(sample[0])))

        if i == 3:
            plt.show()
            break


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
dtype = torch.float32


def zero_initialize(shape) -> torch.Tensor:
    b = torch.zeros(shape, device=device, dtype=dtype)
    b.requires_grad = True
    return b


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for images, labels in loader:
            x = images.to(device=device, dtype=dtype)
            y = labels.to(device=device, dtype=dtype)
            scores = model(x)
            # Arbitrarily selected 1e-3 for now
            # Is this okay? Rounding y -> Either 1.0000 or -1.0000
            # Only if scores >= 0.9995, will we consider answer right 
            num_correct += (abs(scores - torch.round(y,decimals=3))<=1e-2).sum()
            num_samples += scores.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.5f)' % (num_correct, num_samples, 100 * acc))
        return acc


def train_model(model, optimizer, data_loader, val_loader=None, epochs=5):
    model=model.to(device=device)

    acc = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} out of {epochs}")
        for i_batch, (images,labels) in enumerate(data_loader):
            model.train()

            ims = images.to(device=device)
            labs = labels.to(device=device)
            cap = model(ims)
            loss = F.mse_loss(cap, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 10 == 0:
                print('Iteration %d, loss = %.7f' % (i_batch, loss.item()))
        
        if val_loader is not None:
            acc += check_accuracy(val_loader, model)
        print()
    print(f"Average validation accuracy is: {acc*100/i_batch}%")


def test_model(model, data_loader):

    err = 0
    num_correct, num_samples = 0,0
    model.eval()
    with torch.no_grad():
        for i_batch, (images, labels) in data_loader:
            x = images.to(device=device, dtype=dtype)
            y = labels.to(device=device, dtype=dtype)
            scores = model(x)
            err += torch.sum((scores - y)**2)
        err = torch.sqrt(err/len(data_loader))
        num_correct = (abs(scores - torch.round(y,decimals=3))<=1e-2).sum()
        num_samples = len(data_loader)
        acc = float(num_correct) / num_samples
        print(f"RMSE over this cell is: {err}, accuracy is {acc*100}%")
