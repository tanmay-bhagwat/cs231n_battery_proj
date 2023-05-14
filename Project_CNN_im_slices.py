#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
from PIL import Image
from matplotlib.pyplot import imshow
os.chdir(r'/Users/tanmay/Desktop/CS231N/Project/')


# In[2]:


#Convert MATLAB data to dictionary
mat_contents=loadmat('ExampleDC_C1.mat')
data=mat_contents['ExampleDC_C1']['ch'][0][0]
data_cp=data.copy()
dict_vals=[x[:,0] for x in data[0][0]]
dict_keys=list(data.dtype.names)
fin_dict=dict(zip(dict_keys, dict_vals))


# In[3]:


#Convert dictionary to DataFrame
df=pd.DataFrame.from_dict(fin_dict)
#df.to_csv(path_or_buf=r'/Users/tanmay/Desktop/CS231N/Project/ExampleDC1.csv')
df.head()


# In[4]:


df=df.drop(['t','q'],axis=1)


# In[20]:


def make_im(a):
    im_arr=[]
    for key in ['i','v','T']:
        arr=np.array(a[key])
        plot_arr=255*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
        im_arr.append(plot_arr.reshape((15,15)))
        plot_arr=np.dstack(im_arr)
        
    return plot_arr


# In[21]:


#print(df)
frame=df[:225]
a=np.array(frame)
im = Image.fromarray(make_image(a),mode="RGB")
im.show()


# In[6]:


df.keys()


# In[ ]:


#Considering the fact that we allow for overlap between adjacent chunks
#Overlap is a hyperparam here, selected mainly to be able to use all the given datapoints even when len(df)%225!=0
import math
n=225
L=len(df)
print(L)
overlap=75
M=math.floor((L-n)/(n-overlap))+1
print(M)

slices = np.array([j for j in range(0,len(df),225)])-overlap
slices = np.where(slices>0,slices,0)
print(slices)


# In[23]:


j=0
slices = [df.iloc[j:j+225,:] for j in range(0,len(df),225)]
images = []
for i in range(len(slices)-1):
    image=Image.fromarray(make_im(slices[i]),mode='RGB')
    #image.show()
    images.append(image)
images[1].show()


# In[ ]:




