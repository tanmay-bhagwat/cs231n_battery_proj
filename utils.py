import numpy as np
import pandas as pd
import math

# # Introducing overlap between image data

# How do we decide how much overlap to use in two adjacent subintervals?
# Considering the fact that we allow for overlap between adjacent chunks
# Overlap is a hyperparam here, selected mainly to be able to use all the given datapoints even when len(df) % 225 != 0
# 
# We select overlap such that we try to distribute the excess overlap between last and second last panels among all other panels in the full interval and try to equalize overlap between adjacent panels
# 
# ex. Consider the case where L=840, n=225. Typically, we would get only 840//225=3 panels from the whole data
# However, we would like to use the full data as much as possible. Hence, we would like to use the remaining 165 rows as well. This means there will be an overlap of 60 between image 3 and new image 4
# We would like to distribute this excess overlap equally among all other images in the data, such that the overlap between two adjacent images is nearly equal
# 
# Now, we get the overlap by moving the start location of each image to the left, overlapping with the end of the previous image. i.e. (Start of image 2) < 225 (end of image 1). (If there was no overlap, the end of image i would be immediately to the left of start of image i+1).
# 
# So, we want our start of the last image to be at L-n (so as to use full data)
# We currently start at n*/floor(L/n). 
# Hence, excess overlap = n*/floor(L/n) - (L-n)
# We have ceil(L/n) total images (panels), and we can only move ceil(L/n)-2 panels (since start and end panel positions are fixed)
# For ceil(L/n)-2 panels, we have ceil(L/n)-1 overlaps to use!
# Hence, adjacent overlap = (n*/floor(L/n) - (L-n))/(ceil(L/n)-1)

################### Explanation code ###################
# import math
# n=225
# L=len(df)
# print(L)

# init_M=math.floor(L/n)
# print(init_M)
# overlap=math.floor((init_M*n-(L-n))/(math.ceil(L/n)-1)) 
# print(overlap)

# #Using this overlap value, we can now fit in the last set of data to get a new panel, so that total=ceil(L/n)

# indices = np.array([n*j-j*overlap for j in range(0,math.ceil(L/n))])
# indices = np.where(indices>0,indices,0)
# print(indices)
######################################################

def slicer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function takes in DataFrame and returns slices as per overlap explanation above
    Still have to force the last index to be L-n (unclear how to get the correct last index from overlap formula)
    Inputs:
    df : pd.DataFrame
    
    Return:
    slices: Slice view of pd.DataFrame
    """
    L = len(df)
    n = 225
    init_M = math.floor(L/n)
    overlap = math.floor((init_M*n-(L-n))/(math.ceil(L/n)-1)) 
    
    indices = np.array([n*j-j*overlap for j in range(0,math.ceil(L/n))])
    indices = np.where(indices>0,indices,0)
    indices[-1] = L-n
    slices = [df.iloc[j:j+n,:] for j in indices]
    
    return slices


def make_im(df_slice: pd.DataFrame) -> np.ndarray:
    """
    Function returns a normalized (15,15,3) array from given slice of DataFrame
    Order of stacking is 'i','v','T'
    Inputs:
    df_slice: Slice of DataFrame, containing keys 'i','v','T', each of size 225
    
    Return:
    plot_arr: np.ndarray
    """
    im_arr=[]
    for key in ['i','v','T']:
        arr=np.array(df_slice[key])
        plot_arr=255*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
        im_arr.append(plot_arr.reshape((15,15)))
        plot_arr=np.dstack(im_arr)
        
    return plot_arr


def data_loader(mode=0):
    if mode == 0:
        # Working "ExampleDC_C1.mat" MATLAB data to DataFrame
        # Converts all DC_C1.ch data from struct into dict
        mat_contents = loadmat('ExampleDC_C1.mat')
        data = mat_contents['ExampleDC_C1']['ch'][0][0]
        dict_vals = [x[:,0] for x in data[0][0]]
        dict_keys = list(data.dtype.names)
        fin_dict = dict(zip(dict_keys, dict_vals))

        return fin_dict 
    
    if mode == 1:
        # Working with Oxford_Battery_Degradation_Dataset_1.mat
        # Converts all of the Cell{i}.ch data from struct to dict
        mat_contents = loadmat("Oxford_Battery_Degradation_Dataset_1.mat")
        data_dict = {f"Cell{i}":{} for i in range(1,9)}
        for i in range(1,9):
            for cycle in mat_contents[f'Cell{i}'].dtype.names:
                data = mat_contents[f'Cell{i}'][cycle][0][0][0][0][0][0][0]
                dict_vals = [x[:,0] for x in data]
                dict_keys = list(data.dtype.names)
                fin_dict = dict(zip(dict_keys, dict_vals))
                data_dict[f'Cell{i}'][cycle] = pd.DataFrame.from_dict(fin_dict)
        
        return data_dict

