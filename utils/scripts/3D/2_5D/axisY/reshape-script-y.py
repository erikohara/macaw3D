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
from utils.datasets import UKBBT1DatasetOld
from utils.customTransforms import YAxisToFloatUKBB



# In[11]:
# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('y_slice')
args = parser.parse_args()
y_slice = int(args.y_slice)

batch_size = 32
rs = (150,100)
#ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_cropped_slices/T1_cropped_slice_{}/train'.format(z_slice)
ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_cropped_y_slices/T1_cropped_y_slice_{}'.format(y_slice)
output_path = '/work/forkert_lab/erik/MACAW/reshaped/2_5D'


# In[12]:

if not os.path.exists(output_path + '/reshaped_slice_{}_test.npy'.format(y_slice)):


    dataset = UKBBT1DatasetOld(ukbb_path_T1_slices + '/train', transforms.Compose([YAxisToFloatUKBB(),ToTensor(), CenterCrop(rs)]))
    all_loader = DataLoader(dataset, batch_size=batch_size)


    # In[13]:


    data = np.concatenate([d.numpy() for d in all_loader],axis=0)
    data = data.reshape(data.shape[0],-1)

    with open(output_path + '/reshaped_slice_{}.npy'.format(y_slice), 'wb') as f:
        np.save(f, data)

    print("reshaped train set")

    dataset_test = UKBBT1DatasetOld(ukbb_path_T1_slices + '/test', transforms.Compose([YAxisToFloatUKBB(),ToTensor(), CenterCrop(rs)]))
    all_loader_test = DataLoader(dataset_test, batch_size=batch_size)


    # In[13]:


    data_test = np.concatenate([d.numpy() for d in all_loader_test],axis=0)
    data_test = data_test.reshape(data_test.shape[0],-1)

    with open(output_path + '/reshaped_slice_{}_test.npy'.format(y_slice), 'wb') as f2:
        np.save(f2, data_test)

    print("reshaped test set")
