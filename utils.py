import numpy as np
import pandas as pd
import math

def slicer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function takes in DataFrame and returns slices as per overlap explanation above
    Inputs:
    df : pd.DataFrame
    
    Return:
    slices: Slices of DataFrame
    """
    L = len(df)
    n = 225
    init_M=math.floor(L/n)
    overlap=math.floor((init_M*n-(L-n))/(math.ceil(L/n)-1)) 
    
    indices = np.array([n*j-j*overlap for j in range(0,math.ceil(L/n))])
    indices = np.where(indices>0,indices,0)
    print(indices)
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
