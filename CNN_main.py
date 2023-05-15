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


###################################################################
#DATASET SELECTION

# # Working with ExampleDC_C1.mat 
# fin_dict=data_loader()
# #Convert dictionary to DataFrame
# df=pd.DataFrame.from_dict(fin_dict)
# df.head()

# # Dropping all columns not v,T or i
# df=df.drop(['t','q'],axis=1)
# df.head()

# Working with full Oxford dataset
fin_dict = data_loader(mode=1)
df = pd.DataFrame.from_dict(fin_dict['Cell2']['cyc0100'])

# Making a current 'i' column, such that i = d(q)*3600/dt, where dt = 1
j = 0
df['i'] = [0]*len(df)
for j in range(1, len(df)):
  df['i'][j] = (df['q'][j] - df['q'][j-1])*3600    
print(df.head(), type(df['i']))                    
###################################################################

#PREPROCESSING STEPS

# Slicing given dataframe into chunks
slices = slicer(df)

# Converting chunks into images using matplotlib
images = []
i = 0
for i in range(len(slices)):
  image = make_im(slices[i]) #image is a (15,15,3) np array
  images.append(image)
#plt.imshow(images[0])
#plt.show()

# Converting chunks into images using PIL
# i = 0
# for i in range(len(slices)):
#     image = Image.merge("RGB", make_im(slices[i])) #image is an Image object
#     images.append(image)
#     #image.show()

###################################################################
