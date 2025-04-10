#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import torch
import numpy as np


ukbb_path = '/home/erik.ohara/UKBB'
train_files = np.array([])
val_files = np.array([])
test_files = np.array([])
image_folder = '/work/forkert_lab/erik/T1_warped'
cropped_folder = '/work/forkert_lab/erik/T1_cropped_slices'

print("Getting training set")
for each_file in os.listdir(image_folder + '/train'):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        train_files = np.append(train_files,file_id)

print("Getting validation set")
for each_file in os.listdir(image_folder + '/val'):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        val_files = np.append(val_files,file_id)

print("Getting test set")
for each_file in os.listdir(image_folder + '/test'):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        test_files = np.append(test_files,file_id)


print(f"{len(train_files)} files in the train folder")
print(f"{len(val_files)} files in the val folder")
print(f"{len(test_files)} files in thetest folder")


for each_slice in os.listdir(cropped_folder):
    print(f"Starting slice {each_slice}")
    if not os.path.exists(cropped_folder + '/' + each_slice + '/train'):
        os.makedirs(cropped_folder + '/' + each_slice + '/train')
    if not os.path.exists(cropped_folder + '/' + each_slice + '/val'):
        os.makedirs(cropped_folder + '/' + each_slice + '/val')
    if not os.path.exists(cropped_folder + '/' + each_slice + '/test'):
        os.makedirs(cropped_folder + '/' + each_slice + '/test')
    print("Folders created. Starting moving")
    if 'T1_cropped_slice' in each_slice:
        for file_train in train_files:
            if not os.path.exists(cropped_folder + '/' + each_slice + '/train/' + file_train + '.tiff'):
                os.rename(cropped_folder + '/' + each_slice +'/' + file_train + '.tiff', cropped_folder + '/' + each_slice + '/train/' + file_train + '.tiff')
        for file_val in val_files:
            if not os.path.exists(cropped_folder + '/' + each_slice + '/val/' + file_val + '.tiff'):
                os.rename(cropped_folder + '/' + each_slice + '/' + file_val + '.tiff', cropped_folder + '/' + each_slice + '/val/' + file_val + '.tiff')
        for file_test in test_files:
            if not os.path.exists(cropped_folder + '/' + each_slice + '/test/' + file_test + '.tiff'):
                os.rename(cropped_folder  + '/' + each_slice+ '/' + file_test + '.tiff', cropped_folder + '/' + each_slice + '/test/' + file_test + '.tiff')
