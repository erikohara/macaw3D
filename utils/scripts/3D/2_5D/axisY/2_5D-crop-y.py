#!/usr/bin/env python
# coding: utf-8

# In[33]:


import sys
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt

import torchio as tio
import torch

from tqdm.notebook import tqdm

import pickle
from scipy.linalg import eigh 


# In[34]:




# In[11]:
# getting the slice from command line
# 34 to 184
parser = argparse.ArgumentParser()
parser.add_argument('y_slice')
args = parser.parse_args()
y_slice = int(args.y_slice)

ukbb_path_T1_warped = '/work/forkert_lab/erik/T1_warped'
output_path = '/work/forkert_lab/erik/T1_cropped_y_slices/T1_cropped_y_slice_{}'.format(y_slice)

# In[35]:


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)


# In[36]:


subjects_train = []
for file in os.listdir(ukbb_path_T1_warped + '/train'):
        file_name = file.split('.nii')[0]
        subject = tio.Subject(
            name = file_name,
            brain = tio.ScalarImage(os.path.join(ukbb_path_T1_warped,'train', file)),
        )
        subjects_train.append(subject)
print('Train Dataset size:', len(subjects_train), 'subjects')

subjects_test = []
for file in os.listdir(ukbb_path_T1_warped + '/test'):
        file_name = file.split('.nii')[0]
        subject = tio.Subject(
            name = file_name,
            brain = tio.ScalarImage(os.path.join(ukbb_path_T1_warped,'test', file)),
        )
        subjects_test.append(subject)
print('Test Dataset size:', len(subjects_test), 'subjects')

# In[38]:

# 34 to 184
crop_initial = y_slice
crop_end = subjects_train[0].brain.shape[2] - crop_initial - 1
#x_init = round((subjects[0].brain.shape[1]-150)/2)
#x_end = subjects[0].brain.shape[1]- 150 - x_init
#y_init = round((subjects[0].brain.shape[2]-150)/2)
#y_end = subjects[0].brain.shape[2]-150 - y_init
#z_init = round((subjects[0].brain.shape[3]-100)/2)
#z_end = subjects[0].brain.shape[3]-100 - z_init

transform = tio.transforms.Compose([
    tio.transforms.Crop((0,0,crop_initial,crop_end,0,0)),
    #tio.ZNormalization(masking_method=tio.ZNormalization.mean),
],exclude=['name'])


# In[39]:


slice_train_set = tio.SubjectsDataset(
    subjects_train, transform=transform)

slice_test_set = tio.SubjectsDataset(
    subjects_test, transform=transform)

print("Data transformed")

# In[40]:


def save_train_images():
    path_T1_cropped = output_path + '/train'
    if not os.path.exists(path_T1_cropped):
        os.makedirs(path_T1_cropped)
    for individual in slice_train_set:
        individual.brain[:,0,:].save(path_T1_cropped + '/' + individual.name + '.tiff')

def save_test_images():
    path_T1_cropped = output_path + '/test'
    if not os.path.exists(path_T1_cropped):
        os.makedirs(path_T1_cropped)
    for individual in slice_test_set:
        individual.brain[:,0,:].save(path_T1_cropped + '/' + individual.name + '.tiff')


# In[41]:

print("Beggining to save train images")
save_train_images()
print("Beggining to save test images")
save_test_images()


