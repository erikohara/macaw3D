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
parser.add_argument('job_id')
args = parser.parse_args()
job_id = args.job_id

batch_size = 32
rs = 150
#ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_cropped_slices/T1_cropped_slice_{}/train'.format(z_slice)
output_path = '/work/forkert_lab/erik/PCA3D_full'
output_path = f'/scratch/{job_id}'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/full_3D'

# In[12]:


#dataset = UKBBT1Dataset(ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), CenterCrop(rs)]))
#all_loader = DataLoader(dataset, batch_size=batch_size)


# In[13]:


#data = np.concatenate([d.numpy() for d in all_loader],axis=0)
#data = data.reshape(data.shape[0],-1)
#data.shape
data = np.load(reshaped_path + '/reshaped_3D_train.npy')
data = data.reshape(data.shape[0],-1)
print("data loaded and reshaped")

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

with open(output_path + f"/evalues.npy", 'wb') as f_variance:
    np.save(f_variance, np.array(pca.explained_variance_ratio_))

# Get memory size using sys.getsizeof
ecomps = np.array(pca.components_)
memory_size = sys.getsizeof(ecomps)
memory_size_2 = ecomps.nbytes

def format_size(bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

print(memory_size)
print(f"format_size(memory_size): {format_size(memory_size)}")
print(f"format_size(memory_size_2): {format_size(memory_size_2)}")

with open(output_path + f'/evecs.pkl', 'wb') as filefinal:
    pickle.dump(ecomps, filefinal)
    #pickle.dump(pca, filefinal)

with open(output_path + f"/evalues.npy", 'wb') as f_variance:
    np.save(f_variance, np.array(pca.explained_variance_ratio_))

# In[ ]:




