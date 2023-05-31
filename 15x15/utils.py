import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
import os
from Data_Preprocessor import *
import matplotlib.pyplot as plt
import torchvision.transforms as T


# Refer to this: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# Also this: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

class BatteryDataset(torch.utils.data.Dataset):
    """
    Derived class from the abstract dataset class from torch. 
    Retrieves samples as Image object (image) and capacity (label), using annotations file
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        # Because of the way the annotations file was constructed, the first column are just the row numbers.
        # Hence, image filenames are column 2, labels are column 1. 
        img_path = os.path.join(self.root_dir, self.image_dataset.iloc[idx, 2])
        image = read_image(img_path)
        label = self.image_dataset.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
            
        return {'image':image, 'label': label}
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}


def loader(test_cell) -> None:
    """
    Function to call Data_Preprocessor object. Uses DataFrames to write the csv annotations files.

    Parameters:
    test_cell (int): The cell which is the test data
    """
    # Selected cell (test_cell) as the test cell
    # df is then the training dataset DataFrame, can write it to a csv file and use in PyTorch Dataset class
    index_list = [i for i in range(1,9) if i != test_cell]

    obj = Data_Preprocessor('/Users/tanmay/Desktop/CS231N/Project/Code_files/')
    dataset_dict = obj.data_processor()

    # Constructing the training and test annotation files
    df = pd.DataFrame()
    for k in index_list:
        df = pd.concat((df,pd.DataFrame.from_dict(dataset_dict[k])))

    df_test = pd.DataFrame.from_dict(dataset_dict[test_cell])

    # df.to_csv('./image_dir/Training_data.csv')
    # df_test.to_csv('./image_dir/Test_data.csv')

def dataset_visualize(batt_dataset) -> None:
    """Display images from the dataset."""

    for i in range(len(batt_dataset)):
        sample = batt_dataset[i]

        # print(i, sample['image'].shape, sample['label'].shape)
        label = sample['label']

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i})')
        ax.set_xlabel(f'{label:0.5f}')
        # ax.axis('off')
        plt.imshow(np.array(T.ToPILImage()(sample['image'])))

        if i == 3:
            plt.show()
            break
