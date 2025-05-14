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
from utils.datasets import UKBBT1Dataset, CustomDataset
from utils.customTransforms import ToFloatUKBB
from utils.visualize import grid_show, img_grid

from scipy.linalg import eigh 
import pickle
from sklearn.decomposition import PCA


# In[11]:
# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('y_slice')
args = parser.parse_args()
y_slice = int(args.y_slice)

batch_size = 32
rs = 150
#ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_cropped_slices/T1_cropped_slice_{}/train'.format(y_slice)
output_path = '/work/forkert_lab/erik/PCA/slices-y'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-y'

# In[12]:


#dataset = UKBBT1Dataset(ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), CenterCrop(rs)]))
#all_loader = DataLoader(dataset, batch_size=batch_size)


# In[13]:


#data = np.concatenate([d.numpy() for d in all_loader],axis=0)
#data = data.reshape(data.shape[0],-1)
#data.shape
data = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(y_slice))


# In[14]:

'''
cov_matrix = np.matmul(data.T , data)
evalues, evecs = eigh(cov_matrix, eigvals=(0,data.shape[1]-1))
evecs = evecs[::,::-1]
'''
pca = PCA(svd_solver='randomized')
pca.fit(data)

# In[16]:

'''
with open(output_path + '/evecs_slice_{}.pkl'.format(y_slice), 'wb') as file:
    pickle.dump(evecs, file)

with open(output_path + '/evalues_slice_{}.pkl'.format(y_slice), 'wb') as file:
    pickle.dump(evalues, file)
'''

with open(output_path + f"/evalues_slice_{y_slice}.npy", 'wb') as f_variance:
    np.save(f_variance, np.array(pca.explained_variance_ratio_))

with open(output_path + f'/evecs_slice_{y_slice}.pkl', 'wb') as filefinal:
    pickle.dump(pca.components_, filefinal)

# In[ ]:




