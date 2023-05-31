import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import os


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
        # self.transform = transform

    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        # Because of the way the annotations file was constructed, the first column are just the row numbers.
        # Hence, image filenames are column 2, labels are column 1. 
        img_path = os.path.join(self.root_dir, self.image_dataset.iloc[idx, 2])
        image = Image.open(img_path)
        label = np.array([self.image_dataset.iloc[idx, 1]], dtype=np.float32)

        transform = transforms.ToTensor()
        image = transform(image)

        if image.dtype != torch.float32:
            image = image.float()
            
        return (image, label)
