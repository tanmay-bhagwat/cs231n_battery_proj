# -*- coding: utf-8 -*-
"""Images Verification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YL8yjCu7dwg0EtID_jbXXCYpWRP_jwfX
"""

# You may not need this
from google.colab import drive
drive.mount('/content/drive')

from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import os
from PIL import Image, ImageOps

# Change it according to your configuration
os.chdir(r'/content/drive/My Drive/CS_231N/Final Project/Dataset')

"""# Data Loading"""

# Function to create images

def dataToImage_oxford(df1):
  df1=df1.drop(['t','q'],axis=1)
  a=np.array(df1)
  v=a[:,0].copy()
  T=a[:,1].copy()
  i=a[:,2].copy()
  
  plot_v=255*(v-np.min(v))/(np.max(v)-np.min(v))
  plot_v=plot_v.reshape((15,15))
  imv = Image.fromarray(plot_v).convert("L")
 

  plot_T=255*(T-np.min(T))/(np.max(T)-np.min(T))
  plot_T=plot_T.reshape((15,15))

  imT = Image.fromarray(plot_T).convert("L")


  plot_i=255*(i-np.min(i))/(np.max(i)-np.min(i))
  plot_i=plot_i.reshape((15,15))
  #print("i = ", plot_i[0:5,0:5])
  #print("v = ", plot_v[0:5,0:5])
  #print("T = ", plot_T[0:5,0:5])
  imI = Image.fromarray(plot_i).convert("L")
  #imI.show()
  #imv.show()
  #imT.show()

  rgb = Image.merge("RGB", (imI, imv, imT))
  r, g, b = rgb.getpixel((0,0))

  return rgb

"""# Verification"""

mat_contents=loadmat('Oxford_Battery_Degradation_Dataset_1.mat')

# Choose the cell and cycle for verification. This code only verifies the images for C1ch dataset.
# Both the true and our dataset images will be saved in the working directory as 'True Image.png' and 'Test Image.png'.

Cell = 'Cell7'
cyc = 'cyc2100'
data = mat_contents[Cell][cyc][0][0][0][0][0][0][0]
dict_vals=[x[:,0] for x in data]
dict_keys=list(data.dtype.names)
fin_dict=dict(zip(dict_keys, dict_vals))
df=pd.DataFrame.from_dict(fin_dict)

L = len(df)
n = 225
init_M = math.floor(L/n)
overlap = math.floor((init_M*n-(L-n))/(math.ceil(L/n)-1)) 
print("Overlap = ", overlap)

df['i'] = [0]*len(df)
for j in range(1, len(df)):
  df['i'][j] = (df['q'][j] - df['q'][j-1])*3600   

j=1
index = n*j-j*overlap
image = dataToImage_oxford(df[index:index+225])
image.save("True Image.png")
image.show()

image_path = f"/content/drive/MyDrive/CS_231N/Final Project/Dataset/image_dir/{Cell}{cyc}C1chslice{j}.png"  # Update with the correct path to your image
image_dataset = Image.open(image_path)
image_dataset.save("Test Image.png")
image_dataset.show()