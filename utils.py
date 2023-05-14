from scipy.io import loadmat
import pandas as pd
import os

os.chdir(r'/Users/tanmay/Desktop/CS231N/Project/')

mat_contents=loadmat('ExampleDC_C1.mat')
data=mat_contents['ExampleDC_C1']['ch'][0][0]
dict_vals=[x[:,0] for x in data[0][0]]
dict_keys=list(data.dtype.names)
fin_dict=dict(zip(dict_keys, dict_vals))
print(fin_dict.keys())
print(fin_dict['i'])
fin_df=pd.DataFrame(fin_dict)
print(fin_df.head())