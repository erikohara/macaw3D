#!/usr/bin/env python
# coding: utf-8

# In[10]:

import sys
import numpy as np

import matplotlib.pyplot as plt
from  torch.utils.data import Dataset, DataLoader,random_split
from torchvision.transforms import ToTensor, CenterCrop
from torchvision import transforms 
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal

import math
import torch
import torch.distributions as td
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import random 
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import networkx as nx
import yaml

from pathlib import Path
macaw_path = '/home/erik.ohara/macaw'
sys.path.append(macaw_path +'/')
from macaw.flows import Flow, NormalizingFlowModel
from utils.datasets import UKBBT1Dataset, CustomDataset
from utils.customTransforms import ToFloatUKBB
from utils.visualize import grid_show, img_grid
from compression.autoencoder.AE import AE
from utils.helpers import dict2namespace
from macaw import MACAW

import pickle
from tqdm.notebook import tqdm
import pandas as pd


# In[11]:
# getting the slice from command line

slice_initial = 41
slice_final = 141

n_slices = slice_final - slice_initial
nevecs = 50
ncauses = 2
ncomps = 1500
nbasecomps = 25
ukbb_path = '/home/erik.ohara/UKBB'
evec_path = '/work/forkert_lab/erik/PCA/slices-z'
model_path = '/work/forkert_lab/erik/MACAW/models/PCA_hc'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'
output_image = '/work/forkert_lab/erik/MACAW/cf_images/PCA_zero_diff'
original_images = "/work/forkert_lab/erik/T1_warped"
number_images = 2370
cf_age_definition = 55

# In[13]:
print("Starting code")

all_data = []
all_data_dim = []
for z_slice in tqdm(range(slice_initial,slice_final)):
    data = np.load(reshaped_path + '/reshaped_test_slice_{}.npy'.format(z_slice))
    all_data.append(data)
    all_data_dim.append(data.shape[1])

# loading train data
'''
all_data_train = []
for z_slice in tqdm(range(slice_initial,slice_final)):
    data = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(z_slice))
    all_data_train.append(data)
'''

all_evecs = []
for z_slice in range(slice_initial,slice_final):
    with open(evec_path + '/evecs_slice_{}.pkl'.format(z_slice),'rb') as f:  
        evecs = pickle.load(f)
        all_evecs.append(evecs)
# In[14]:


def encode(data, evecs):
    return np.matmul(data,evecs)

def decode(data,evecs):
    return np.matmul(data,evecs.T)

# In[16]:

data_path = ukbb_path + '/ukbb_img.csv'

df = pd.read_csv(data_path,low_memory=False)
print(f"The original size of the dataframe is {df.shape}")
# adding noise to sex and normalizing age
#df['Sex'] = df['Sex'] + np.random.rand(df.shape[0]) - 0.5
#scaler_a = StandardScaler()
#df['Age'] = scaler_a.fit_transform(df[['Age']])

# getting the only the test set
df_test = pd.DataFrame(columns=df.columns)
# test folder
for each_file in os.listdir(original_images + '/test'):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        df_test = pd.concat([df[df['eid'] == int(file_id)],df_test.loc[:]]).reset_index(drop=True)
df_test.sort_values(by=['eid'], inplace=True)

df = df_test
print(f"The size of the dataframe (just test set) is {df.shape}")


all_eid = df[['eid']].to_numpy()
#causes = df[['Age','Sex']].to_numpy()
min_age = df['Age'].min()
print(f"Age min: {min_age}")
sex = df['Sex'] 
age = df['Age'] - min_age

with open(macaw_path + '/config/ukbb.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
    
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
'''
all_scalers = []
for z_slice in tqdm(range(slice_initial,slice_final)):
    order_slice = z_slice - slice_initial
    scalers = {}
    for e in range(0,ncomps,nevecs):
        encoded_data =  encode(all_data_train[order_slice],all_evecs[order_slice][:,e:e+nevecs])
        scaler = StandardScaler()
        encoded_data = scaler.fit_transform(encoded_data)
        scalers[f"{e}"] = scaler
    all_scalers.append(scalers)
'''
    
cf_vals = {1:cf_age_definition - min_age}
nsamples = number_images

rands = np.random.randint(0,all_data[0].shape[0],nsamples)    
#c_obs = causes[rands,:]
sex_obs = sex[rands]
age_obs = age[rands]
random_eids = all_eid[rands,0]

all_d_obs = []

for z_slice in range(slice_initial,slice_final):
    order_slice = z_slice - slice_initial
    all_d_obs.append(all_data[order_slice][rands,:])

#all_d_encodes = []
all_s = []
all_c_cf = [np.zeros((nsamples,ncomps)) for _ in range(slice_initial,slice_final)]
for z_slice in tqdm(range(slice_initial,slice_final)):
    order_slice = z_slice - slice_initial
    #d_encodes = []
    s=0
    for ev in range(0,ncomps-nbasecomps,nevecs-nbasecomps):
    #for ev in range(0,all_data_dim[order_slice],nevecs):
        encoded_obs =  encode(all_d_obs[order_slice],all_evecs[order_slice][:,ev:ev+nevecs])   
        try:
            '''
            macaw = MACAW.MACAW(config)
            print("Iniciando modelo")
            macaw.model = NormalizingFlowModel([],[Flow(nevecs + 4,[(1,2)],config.device)]).to(config.device)
            print("Iniciando carregamento")
            macaw.model.load_state_dict(torch.load(model_path + f'/slice_{z_slice}/{nevecs}/macaw_ukbb_PCA_{ev}_cpu.pt',
                                                    map_location=config.device), strict=False)
            '''
            macaw = torch.load(model_path + f'/slice_{z_slice}/macaw_ukbb_PCA_{ev}.pt')
            #scaler = all_scalers[order_slice][f"{ev}"]
            #encoded_obs = scaler.transform(encoded_obs)
            #max_value = encoded_obs.max()
            #min_value = encoded_obs.min()
            #X_obs = np.hstack([c_obs,encoded_obs])  
            macaw.model.eval()
            with torch.no_grad():
                X_obs = np.hstack([sex_obs[:,np.newaxis], age_obs[:,np.newaxis], encoded_obs]) 
                z_obs = macaw._forward_flow(X_obs)
                cc = macaw._backward_flow(z_obs)
            #d_encoded_cf = cf[:,ncauses:]
            # limiting problems with infinitive values
            #d_encoded_cf[d_encoded_cf>max_value] = max_value
            #d_encoded_cf[d_encoded_cf<min_value] = min_value
            #c_cf = cf[:,:ncauses]
            #d_encoded_cf = scaler.inverse_transform(d_encoded_cf)
        except FileNotFoundError as e:
            cc[:,ncauses:] = X_obs[:,ncauses:].copy()
            #d_encoded_cf = encoded_obs
        '''
        except ValueError as value_error:
            print(f"IDs of subjects: {random_eids}")
            print(d_encoded_cf.shape)
            print(np.isinf(d_encoded_cf).nonzero())
            print(f"Error in slice {z_slice}")
            print(f"in file /slice_{z_slice}/{nevecs}/macaw_ukbb_PCA_{ev}.pt")
            raise Exception('Just to stop and see', value_error)
        '''
        all_c_cf[order_slice][:,ev:ev+nevecs] = cc[:,ncauses:]
        #d_encodes.append(d_encoded_cf)
        s+=1
    #all_d_encodes.append(d_encodes)
    all_s.append(s)

        
#age_obs = age_obs + min_age
#sex = ['Male' if round(s) else 'Female' for s in age_obs]
#titles = [f'Age:{a}, Sex:{s}' for a,s in zip(age_obs,sex)]


sex_cf = ['Male' if round(s) else 'Female' for s in cc[:,0]]
age_cf = cc[:,1] + min_age
titles_cf = [f'Age:{np.round(a)}, Sex:{s}' for a,s in zip(age_cf,sex_cf)]
        
all_decoded_cf = []
for z_slice in tqdm(range(slice_initial,slice_final)):
    order_slice = z_slice - slice_initial
    #encoded_cf = np.hstack(all_d_encodes[order_slice])
    decoded_cf = decode(all_c_cf[order_slice],all_evecs[order_slice][:,:ncomps])
    all_decoded_cf.append(decoded_cf)
    
import nibabel as nib


for individual in range(nsamples):
    numpy_image = np.array([])
    for z_slice in range(slice_initial,slice_final):
        order_slice = z_slice - slice_initial
        if order_slice == 0:
            numpy_image = all_decoded_cf[order_slice][individual].reshape(150,150)
        else:
            numpy_image = np.dstack((numpy_image,all_decoded_cf[order_slice][individual].reshape(150,150)))
    #numpy_image = np.swapaxes(numpy_image,0,1)
    # Getting the orifinal affine and header
    image_path = original_images + '/test/' + str(random_eids[individual]) + '.nii.gz'
    original_image = nib.load(image_path)
    # Saving new generated images
    ni_img = nib.Nifti1Image(numpy_image, original_image.affine, original_image.header)
    nib.save(ni_img, output_image + "/" + str(random_eids[individual])  + f"_{slice_initial}-{slice_final-1}.nii.gz")
