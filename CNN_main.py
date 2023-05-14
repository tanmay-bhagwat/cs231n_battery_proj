#!/usr/bin/env python
# coding: utf-8

from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
from PIL import Image
from matplotlib.pyplot import imshow
from utils import *
os.chdir(r'/Users/tanmay/Desktop/CS231N/Project/')


# # Working with ExampleDC_C1.mat 
# fin_dict=data_loader()
# #Convert dictionary to DataFrame
# df=pd.DataFrame.from_dict(fin_dict)
# df.head()

# # Dropping all columns not v,T or i
# df=df.drop(['t','q'],axis=1)
# df.head()

# slices = slicer(df)
# images = []
# for i in range(len(slices)):
#     image=Image.fromarray(make_im(slices[i]),mode='RGB')
#     #image.show()
#     images.append(image)
# #images[0].show()


# Working with full Oxford dataset
fin_dict = data_loader(mode=1)
df = pd.DataFrame.from_dict(fin_dict['Cell2']['cyc0100'])

# Making a current 'i' column, such that i = d(q)*3600/dt, where dt = 1
j = 0
df['i'] = [0]*len(df)
for j in range(1, len(df)):
  df['i'][j] = (df['q'][j] - df['q'][j-1])*3600    
print(df.head(), type(df['i']))                    

# Slicing given dataframe into chunks
slices = slicer(df)
#print(fin_dict['Cell2']['cyc0100'].shape)
#print(len(slices))

# Converting chunks into images
images = []
i = 0
for i in range(len(slices)):
    image = Image.fromarray(make_im(slices[i]),mode='RGB')
    images.append(image)
#print(len(images))
