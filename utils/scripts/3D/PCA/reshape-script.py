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
from utils.datasets import UKBBT1DatasetOld, CustomDataset
from utils.customTransforms import ToFloatUKBB
from utils.visualize import grid_show, img_grid

from scipy.linalg import eigh 
import pickle


# In[11]:
# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('z_slice')
args = parser.parse_args()
z_slice = int(args.z_slice)

batch_size = 32
rs = 150
ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_cropped_slices/T1_cropped_slice_{}/train'.format(z_slice)
output_path = '/work/forkert_lab/erik/MACAW/reshaped/PCA'


# In[12]:


dataset = UKBBT1DatasetOld(ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), CenterCrop(rs)]))
all_loader = DataLoader(dataset, batch_size=batch_size)


# In[13]:


data = np.concatenate([d.numpy() for d in all_loader],axis=0)
data = data.reshape(data.shape[0],-1)

with open(output_path + '/reshaped_slice_{}.npy'.format(z_slice), 'wb') as f:
    np.save(f, data)
