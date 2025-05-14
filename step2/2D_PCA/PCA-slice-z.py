#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
import numpy as np
import argparse

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from pathlib import Path
sys.path.append('/home/erik.ohara/macaw/')
import pickle
from sklearn.decomposition import PCA

# In[11]:
# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('z_slice')
args = parser.parse_args()
z_slice = int(args.z_slice)

batch_size = 32
rs = 150
#ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_cropped_slices/T1_cropped_slice_{}/train'.format(z_slice)
output_path = '/work/forkert_lab/erik/PCA/slices-z-sklearn'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'

# In[12]:


#dataset = UKBBT1Dataset(ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), CenterCrop(rs)]))
#all_loader = DataLoader(dataset, batch_size=batch_size)


# In[13]:


#data = np.concatenate([d.numpy() for d in all_loader],axis=0)
#data = data.reshape(data.shape[0],-1)
#data.shape
data = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(z_slice))


# In[14]:

'''
cov_matrix = np.matmul(data.T , data)
evalues, evecs = eigh(cov_matrix, eigvals=(0,data.shape[1]-1))
evecs = evecs[::,::-1]
'''

pca = PCA(svd_solver='randomized')
pca.fit(data)

#evecs = np.matmul(all_evecs.T , all_evecs)
#print("cov_matrix calcuated")
#_, evecs = eigh(evecs, eigvals=(0,all_evecs.shape[1]-1))
#print("evec calculated")
#evecs = evecs[::,::-1]

with open(output_path + f"/evalues_slice_{z_slice}.npy", 'wb') as f_variance:
    np.save(f_variance, np.array(pca.explained_variance_ratio_))

with open(output_path + f'/evecs_slice_{z_slice}.pkl', 'wb') as filefinal:
    pickle.dump(pca.components_, filefinal)

# In[ ]:




