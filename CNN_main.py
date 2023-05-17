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

###################################################################

#CURRENT COLUMN ADDITION

# Making a current 'i' column, such that i = d(q)*3600/dt, where dt = 1
j = 0
df['i'] = [0]*len(df)
for j in range(1, len(df)):
  df.loc[df.index[j],'i'] = (df.loc[df.index[j],'q'] - df.loc[df.index[j-1],'q'])*3600
  
###################################################################

#PREPROCESSING STEPS

# Slicing given dataframe into chunks
slices = slicer(df)

# Converting chunks into images using matplotlib, not used since several functions are internally using PIL directly or indirectly
# Conversion from np array to Image object causes some changes in pixel values
# images = []
# i = 0
# for i in range(len(slices)):
#   image = make_im(slices[i]) #image is a (15,15,3) np array
#   images.append(image)
#plt.imshow(images[0])
#plt.show()

i = 0
images = []
for i in range(len(slices)):
    image = Image.merge("RGB", make_im_Image(slices[i]))
    images.append(image)
    #image.show()
#images[0].show()
#print(im)

###################################################################
