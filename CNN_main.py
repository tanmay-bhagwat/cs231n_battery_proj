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


# # Converting MATLAB data to DataFrame

#Convert MATLAB data to dictionary
mat_contents=loadmat('ExampleDC_C1.mat')
data=mat_contents['ExampleDC_C1']['ch'][0][0]
data_cp=data.copy()
dict_vals=[x[:,0] for x in data[0][0]]
dict_keys=list(data.dtype.names)
fin_dict=dict(zip(dict_keys, dict_vals))

#Convert dictionary to DataFrame
df=pd.DataFrame.from_dict(fin_dict)
df.head()

# Dropping all columns not v,T or i
df=df.drop(['t','q'],axis=1)
df.head()


slices = slicer(df)
images = []
for i in range(len(slices)):
    image=Image.fromarray(make_im(slices[i]),mode='RGB')
    #image.show()
    images.append(image)
images[0].show()
