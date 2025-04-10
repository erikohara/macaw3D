#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
import numpy as np
import pandas as pd
import argparse

from  torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, CenterCrop
from torchvision import transforms 

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from pathlib import Path
sys.path.append('/home/erik.ohara/macaw/')
from utils.datasets import UKBBT13DDataset, CustomDataset
from utils.customTransforms import ToFloatUKBB, Crop3D
from utils.visualize import grid_show, img_grid

from scipy.linalg import eigh 
import pickle


# In[11]:
# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('cf_path')
parser.add_argument('axis_reshape')
args = parser.parse_args()
cf_path = args.cf_path
# which axis to reshape for (z=1, x=2, y=3)
axis_reshape = int(args.axis_reshape)



batch_size = 32
crop_size = (100,150,150)
#ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/val'
ukbb_path_T1_slices = f'/work/forkert_lab/erik/MACAW/cf_images/{cf_path}'
output_path = f'/work/forkert_lab/erik/MACAW/reshaped/{cf_path}'
#output_path = f'/work/forkert_lab/erik/MACAW/reshaped'
df_ukbb= '/home/erik.ohara/UKBB/test.csv'
df= pd.read_csv(df_ukbb,low_memory=False)
df.sort_values(by=['eid'], inplace=True)

dataset = UKBBT13DDataset(df,ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# In[13]:


# which axis to reshape for (z=1, x=2, y=3)
#axis_reshape = 1
#data = np.concatenate([d.numpy() for d in all_loader],axis=0)
data = np.concatenate([d.numpy() for d,_,_,_,_ in all_loader],axis=0)
print(data.shape)

for n_slice in range(data.shape[axis_reshape+1]):
    if axis_reshape == 3:
        folder = '/slices-y'
        initial_slice = 34
        data_slice = data[:,:,:,:,n_slice].reshape(data.shape[0],-1)
    elif axis_reshape == 2:
        folder = '/slices-x'
        initial_slice = 16
        data_slice = data[:,:,:,n_slice,:].reshape(data.shape[0],-1)
    else:
        folder = '/slices-z'
        initial_slice = 41
        data_slice = data[:,:,n_slice,:,:].reshape(data.shape[0],-1)
    if not os.path.exists(output_path + folder):
        os.makedirs(output_path + folder)
    with open(output_path + folder + '/reshaped_test_slice_{}.npy'.format(initial_slice + n_slice), 'wb') as f:
        np.save(f, data_slice)
