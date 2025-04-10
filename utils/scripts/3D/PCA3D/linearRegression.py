import torch
import sys
import pickle
import nibabel as nib
import pandas as pd
import numpy as np

from  torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

sys.path.append('/home/erik.ohara/macaw/')
from utils.customTransforms import ToFloatUKBB, Crop3D
from utils.datasets import UKBBT13DDataset, CustomDataset
from sklearn.linear_model import LinearRegression

batch_size = 4
crop_size = (100,150,150)
n_sample = 8
nevecs = 1500

ukbb_path_T1_train = '/work/forkert_lab/erik/T1_warped/train'
df_ukbb_train = '/home/erik.ohara/UKBB/train.csv'
ukbb_path_T1_val = '/work/forkert_lab/erik/T1_warped/val'
df_ukbb_val = '/home/erik.ohara/UKBB/val.csv'
evec_path = '/work/forkert_lab/erik/PCA3D'
#output_path = '/work/forkert_lab/erik/MACAW/decoded_images/PCA3D_15000'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# random sample to test
df_train = pd.read_csv(df_ukbb_train,low_memory=False)
df_val = pd.read_csv(df_ukbb_val,low_memory=False)

dataset_train= UKBBT13DDataset(df_train,ukbb_path_T1_train, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
train_loader = DataLoader(dataset_train, batch_size=batch_size)

dataset_val= UKBBT13DDataset(df_val,ukbb_path_T1_val, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
val_loader = DataLoader(dataset_val, batch_size=batch_size)

with open(evec_path + "/evecs.pkl",'rb') as f:  
    evecs3D = pickle.load(f)
print("PCA loaded")


image_data_train = []
ages_train = []
data_train = [(d.numpy(),age) for d,_,age,_,_ in train_loader]
for each_data in data_train:
    image_data_train.append(each_data[0])
    ages_train.append(each_data[1])
image_data_train = np.concatenate(image_data_train,axis=0)
#eids_train = np.concatenate(eids_train,axis=0)
print(f"image_data_train.shape: {image_data_train.shape}")

image_data_train = image_data_train.reshape(image_data_train.shape[0],-1)
print(image_data_train.shape)

def encode(data, evecs):
    return np.matmul(data,evecs.T)

def decode(data,evecs):
    return np.matmul(data,evecs)

encoded_data_train = encode(image_data_train,evecs3D[:nevecs])  

#print(eids)

reg = LinearRegression().fit(encoded_data_train, ages_train)

image_data_val= []
ages_val = []
data_val = [(d.numpy(),age) for d,_,age,_,_ in val_loader]
for each_data in data_val:
    image_data_val.append(each_data[0])
    ages_val.append(each_data[1])
image_data_val = np.concatenate(image_data_val,axis=0)
print(f"image_data_val.shape: {image_data_val.shape}")

image_data_val = image_data_val.reshape(image_data_val.shape[0],-1)
print(image_data_val.shape)

encoded_data_val= encode(image_data_val,evecs3D[:nevecs])  

predictions = reg.predict(encoded_data_val)

loss_func = torch.nn.L1Loss()

MAE_loss = loss_func(predictions, ages_val)

print(f"MAE_loss: {MAE_loss}")