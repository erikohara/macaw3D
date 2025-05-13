#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
import numpy as np
import argparse

from  torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, CenterCrop
from torchvision import transforms 

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from pathlib import Path
sys.path.append('/home/erik.ohara/macaw/')
from utils.datasets import UKBBT13DDataset
from utils.customTransforms import ToFloatUKBB, Crop3D
import pandas as pd


# In[11]:
# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('cf_path')
parser.add_argument('original_job_id')
parser.add_argument('job_id')
args = parser.parse_args()
cf_path = args.cf_path
original_job_id = args.original_job_id
job_id = args.job_id
'''
'''

batch_size = 32
crop_size = (180,180,200)
#crop_size = (102,150,150)
#ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/train'
ukbb_path_T1_slices = f'/work/forkert_lab/erik/MACAW/cf_images/{cf_path}'
#ukbb_path_T1_slices = f'/scratch/{original_job_id}/{cf_path}'
#output_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
#output_path = '/work/forkert_lab/erik/MACAW/reshaped/full_3D'
output_path = f'/scratch/{job_id}'
df_ukbb= '/home/erik.ohara/UKBB/test.csv'

# In[12]:

df= pd.read_csv(df_ukbb,low_memory=False)

dataset = UKBBT13DDataset(df,ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
#dataset = UKBBT13DDataset(df,ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor()]))
#dataset = UKBBT13DDataset(ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# In[13]:

# which axis to reshape for (z=1, x=2, y=3)
#axis_reshape = 2
data = np.concatenate([d.numpy() for d,_,_,_,_ in all_loader],axis=0)
print(data.shape)


with open(output_path + f'/reshaped_3D_{cf_path}.npy', 'wb') as f:
    np.save(f, data)
