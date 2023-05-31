#!/usr/bin/env python
# coding: utf-8

import math
import json
import os
import numpy as np
import pandas as pd
from PIL import Image


class Data_Preprocessor():
    """
    Handles data preprocessing operations. Combines the operations of insertion of 'current' columns, slicing data and
    generating images for use. Stores images in './image_dir' directory.

    Parameters:
    path (str): Path to the directory where the json files are stored\\
    data_select (int, optional): To select whether to use the smaller 'ExampleDC_C1' data or the full Oxford dataset
                                (default = full Oxford dataset)
    transform (Transform object): Any transformations that need to be applied
    """

    def __init__(self, path, data_select=1, transform=None) -> None:
        self.data_select = data_select
        self.cell_dict = {i:pd.DataFrame() for i in range(1,9)}
        self.path = path
        self.transform = transform


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
    #######################################################

    def _slicer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function takes in DataFrame and returns slices as per overlap explanation above
        Still have to force the last index to be L-n (unclear how to get the correct last index from overlap formula)
        n is the number of datapoints per chunk (make sure to use perfect squares), L is the total number of datapoints in the 
        charge/ discharge cycle, M is the number of chunks generated and c is the overlap.

        Parameters:
        df : pd.DataFrame, the DataFrame to be sliced
        
        Return:
        slices: pd.DataFrame, slices of DataFrame
        """
        L = len(df)
        n = 225
        init_M = math.floor(L/n)
        overlap = math.floor((init_M*n-(L-n))/(math.ceil(L/n)-1)) 
        
        indices = np.array([n*j-j*overlap for j in range(0,math.ceil(L/n))])
        indices = np.where(indices>0,indices,0)
        indices[-1] = L-n
        #print(indices)
        slices = [df.iloc[j:j+n,:] for j in indices]
        
        return slices

    def _make_im_Image(self, df_slice: pd.DataFrame) -> np.ndarray:
        """
        Function returns a list of 3 normalized (15,15) Image objects from given slice of DataFrame
        Order of stacking is 'i','v','T'

        Parameters:
        df_slice: Slice of DataFrame, containing keys 'i','v','T', each of size 225
        
        Return:
        im_arr: list, containing Image objects
        """
        im_arr=[]
        for key in ['i','v','T']:
            arr = np.array(df_slice[key])
            plot_arr = 255*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
            im_arr.append(Image.fromarray(plot_arr.reshape((15,15))).convert('L'))
            
        return im_arr
    
    def _data_loader(self) -> None:
        """
        Uses json files to convert to Python dicts. For MATLAB, can use jsonencode function to convert mat files to json.
        Calculates and adds 'current' column to every charge and discharge cycle DataFrame in the full Oxford data.
        Checks for proper addition of this column and for column length mismatch with 'v' column in the dataset (could use any other 
        existing column as well).
        Updates cell_dict param with these DataFrames.
        """
        
        # if self.mode == 0:
        #     # Working "ExampleDC_C1.mat" MATLAB data to DataFrame
        #     with open('./file.json') as f:
        #         fin_dict = json.load(f)

        #     return fin_dict 
        
        if self.data_select == 1:
            # Working with full Oxford dataset
            # Preprocessing data, converting from MATLAB to dict
            for i in range(1,9):
                dataset_loc = os.path.join(self.path)
                os.chdir(dataset_loc)
                with open(f'./Cell_{i}.json') as f:
                    fin_dict = json.load(f)
                read_df = pd.DataFrame.from_dict(fin_dict)
                for cycle_num in read_df.keys():
                    for cycle_part in read_df[cycle_num].keys():
                         # if "C1" not in cycle_part:
                         #     read_df[cycle_num].drop(cycle_part, inplace=True)
                        
                        # Adding a current column; since q is given in mAh (see README.txt), current is in mA
                        read_df[cycle_num][cycle_part]['i'] = np.zeros(len(read_df[cycle_num][cycle_part]['q']))
                        for key in read_df[cycle_num][cycle_part].keys():
                            if key == 'q':
                                read_df[cycle_num][cycle_part][key] = np.array(read_df[cycle_num][cycle_part][key]).reshape(1,-1).T
                                ser = np.concatenate([np.zeros((1,1)), read_df[cycle_num][cycle_part][key][:-1]])
                                read_df[cycle_num][cycle_part]['i'] = 3600*(read_df[cycle_num][cycle_part][key] - ser).reshape(-1)
                                read_df[cycle_num][cycle_part][key] = read_df[cycle_num][cycle_part][key].reshape(-1)

                        # Checking to make sure every dict has a current 'i' key
                        if 'i' not in read_df[cycle_num][cycle_part].keys():
                            raise KeyError('No key i found')
                        # Checking to make sure every current np array has the same length as other np arrays in the dict 
                        if len(read_df[cycle_num][cycle_part]['i']) != len(read_df[cycle_num][cycle_part]['v']):
                            raise ValueError(f"List length mismatch, for i: {len(read_df[cycle_num][cycle_part]['i'])}, for v: {len(read_df[cycle_num][cycle_part]['v'])}")

                self.cell_dict[i] = read_df

    def data_processor(self) -> dict:
        """
        Generates the images to be used in networks as well as the annotations file as a csv. The csv is structured as: 'image' column
        with names as '{Cell number}{cycle number}{cycle part}{slice number}.png' and the 'label' column as the capacity over that slice number.
        """

        self._data_loader()
        # Making a 'dataset dict' which collects all the images and labels per cell
        i = 0 
        m = 0

        dataset_dict = {m:{} for m in range(1,9)}
        i = 0 
        for i in range(1,9):
            dataset_dict[i]['labels']=[]
            dataset_dict[i]['images']=[]
            for cycle_num in self.cell_dict[i].keys():
                cycle_part = 0
                for cycle_part in self.cell_dict[i][cycle_num].keys():

                    # Performing the slicing operation to get images
                    # Store images in 'image_dir' directory
                    df = pd.DataFrame.from_dict(self.cell_dict[i][cycle_num][cycle_part])
                    slices = self._slicer(df)
                    j = 0
                    for j in range(len(slices)):
                        image = Image.merge("RGB", self._make_im_Image(slices[j]))
                        image.save(f'./image_dir/Cell{i}{cycle_num}{cycle_part}slice{j}.png')
                        dataset_dict[i]['labels'].append(np.mean(slices[j]['q']))
                        dataset_dict[i]['images'].append(f'Cell{i}{cycle_num}{cycle_part}slice{j}.png')
                    
                    # To work as a progress bar
                    print(i, cycle_num, cycle_part, len(dataset_dict[i]['images']), len(dataset_dict[i]['labels']), len(dataset_dict[i]['images'])==len(dataset_dict[i]['labels']))
                    
                    # Checking for equality between image list length and label list length
                    if len(dataset_dict[i]['images'])!=len(dataset_dict[i]['labels']):
                        raise ValueError(f"Length mismatch for {i}, {cycle_num}, {cycle_part}")

        # Checking to ensure that every cell dictionary has one (15,15,3) matrix as the 'image' element and one label for that element
        for _ in range(1000):
            i = np.random.randint(1,9)
            if _%100 == 0:
                    print(f"In cell {i}")
            cell = pd.DataFrame.from_dict(dataset_dict[i])
            count = 0
            for count in range(1000):
                j = np.random.randint(0, len(cell['images']))
                if len([cell.iloc[j,0]]) != 1:
                    raise ValueError("More than one image stored for a label")
                
        return dataset_dict

        # Final structure of the cell_dict is:
        # cell_dict[i] -> pd.DataFrame
        # cell_dict[i][cycle_num] -> pd.Series
        # cell_dict[i][cycle_num][cycle_part] -> dict
        # cell_dict[i][cycle_num][cycle_part][key] -> np.ndarray
