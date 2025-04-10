#!/usr/bin/env python
# coding: utf-8

# In[10]:

import sys
import numpy as np

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
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
from compression.vqvae import vqvae
from utils.helpers import dict2namespace
from macaw import MACAW

import pickle
from tqdm.notebook import tqdm
import pandas as pd
import argparse

# In[11]:
# getting the times of repetitions from command line
parser = argparse.ArgumentParser()
parser.add_argument('repeat')
args = parser.parse_args()
repeat = int(args.repeat)

nevecs = 50
ncauses = 2
ncomps = 10625
nbasecomps = 25
nchannels = 8
ukbb_path = '/home/erik.ohara/UKBB'
model_path = f"/work/forkert_lab/erik/MACAW/models/macaw_vqvae8_50_nevecs"
scalers_path = '/work/forkert_lab/erik/MACAW/scalers/macaw_vqvae8_50_nevecs'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
output_image = f'/work/forkert_lab/erik/MACAW/cf_images/macaw_vqvae8_50_nevecs_zero_repeat_{repeat}'
cf_age_path = '/work/forkert_lab/erik/MACAW/2_5'
original_images = "/work/forkert_lab/erik/T1_warped"
vqvae_path = '/work/forkert_lab/erik/MACAW/models/vqvae3D_8'
number_images = 2370

# In[13]:
print("Starting code")
if not os.path.exists(output_image):
    os.makedirs(output_image)

data = np.load(reshaped_path + '/reshaped_3D_102_150_150_test.npy')
#data = data.reshape(data.shape[0],-1)
print("Data loaded")

# loading train data
'''
all_data_train = []
for z_slice in tqdm(range(slice_initial,slice_final)):
    data = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(z_slice))
    all_data_train.append(data)
with open(evec_path + "/evecs.pkl",'rb') as f:  
    evecs3D = pickle.load(f)
print("PCA loaded")
'''
# Priors
# Getting Age and Sex data
data_all_path = ukbb_path + '/ukbb_img.csv'
df_all = pd.read_csv(data_all_path,low_memory=False)
min_age = df_all['Age'].min()

sex = df_all['Sex'] 
age = df_all['Age'] - min_age
P_sex = np.sum(sex)/len(sex)

with open(macaw_path + '/config/ukbbVQVAE.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
    
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

unique_values, counts = np.unique(age, return_counts=True)
P_age = counts/np.sum(counts)
priors = [(slice(0,1),td.Bernoulli(torch.tensor([P_sex]).to(config.device))), # sex
          (slice(1,2),td.Categorical(torch.tensor([P_age]).to(config.device))), # age
          (slice(ncauses,(nbasecomps*nchannels)+ncauses),td.Normal(torch.zeros(nbasecomps*nchannels).to(config.device), torch.ones(nbasecomps*nchannels).to(config.device))), # base_comps
          (slice((nbasecomps*nchannels)+ncauses,(nevecs*nchannels)+ncauses),td.Normal(torch.zeros((nevecs-nbasecomps)*nchannels).to(config.device), torch.ones((nevecs-nbasecomps)*nchannels).to(config.device))), # new_comps
         ]

# causal Graph
sex_to_latents = [(0,i) for i in range(ncauses,nevecs*nchannels+ncauses)]
age_to_latents = [(1,i) for i in range(ncauses,nevecs*nchannels+ncauses)]
autoregressive_latents = [(i,j) for i in range(ncauses,nevecs*nchannels+ncauses-1) for j in range(i+1,nevecs*nchannels+ncauses)]
edges = sex_to_latents + age_to_latents + autoregressive_latents


# In[16]:

data_path = ukbb_path + '/test.csv'

df = pd.read_csv(data_path,low_memory=False)
#df.sort_values(by=['eid'], inplace=True)
print(f"The original size of the dataframe is {df.shape}")
# adding noise to sex and normalizing age
#df['Sex'] = df['Sex'] + np.random.rand(df.shape[0]) - 0.5
#scaler_a = StandardScaler()
#df['Age'] = scaler_a.fit_transform(df[['Age']])

# getting the only the test set
'''
df_test = pd.DataFrame(columns=df.columns)
# test folder
for each_file in os.listdir(original_images + '/test'):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        df_test = pd.concat([df[df['eid'] == int(file_id)],df_test.loc[:]]).reset_index(drop=True)
df_test.sort_values(by=['eid'], inplace=True)

df = df_test
print(f"The size of the dataframe (just test set) is {df.shape}")
'''


all_eid = df[['eid']].to_numpy()
#causes = df[['Age','Sex']].to_numpy()
min_age = df['Age'].min()
print(f"Age min: {min_age}")
sex = df['Sex'] 
age = df['Age'] - min_age


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
    
#cf_vals = {1:cf_age_definition - min_age}


nsamples = number_images

#rands = np.random.randint(0,all_data[0].shape[0],nsamples)    
#c_obs = causes[rands,:]
sex_obs = sex
age_obs = age
random_eids = all_eid[:,0]

# create list with +5 and -5 for the countefactual age
# create list with +5 and -5 for the countefactual age
cf_age_obs = age_obs
'''
with open(cf_age_path + f'/cf_age_obs_2.pkl','rb') as cfagefile:  
    cf_age_obs = pickle.load(cfagefile)
minus_5 = 0
plus_5 = 0
cf_age_obs = np.copy(age_obs)
for idx,each_age in enumerate(cf_age_obs):
    if each_age < 5:
        cf_age_obs[idx] = cf_age_obs[idx] + 5
        plus_5 += 1
    elif each_age>30:
        cf_age_obs[idx] = cf_age_obs[idx] - 5
        minus_5 += 1
    else:
        rand_plus_minus = np.random.choice([-5,5], 1)
        if rand_plus_minus[0] == 5:
            cf_age_obs[idx] = cf_age_obs[idx] + 5
            plus_5 += 1
        else:
            cf_age_obs[idx] = cf_age_obs[idx] - 5
            minus_5 += 1
print(f"The quantity of plus 5 is: {plus_5}")
print(f"The quantity of minus 5 is: {minus_5}")
#diff_5 = np.random.choice([-5,5], number_images, replace=True)
with open(cf_age_path + f'/cf_age_obs_2.pkl', 'wb') as cfagefile:
    pickle.dump(cf_age_obs, cfagefile)
'''

#residuals = data - decode(encode(data,evecs3D[:ncomps]),evecs3D[:ncomps])
#all_d_encodes = []
# loading VQVAE
with open(macaw_path +'/compression/vqvae/vqvae.yaml', 'r') as f:
    config_raw_vq_vae = yaml.load(f, Loader=yaml.FullLoader)
config_vq_vae = dict2namespace(config_raw_vq_vae)
#config_vq_vae.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config_vq_vae.device = torch.device('cpu')

writer = SummaryWriter(macaw_path + 'logs/vqvae_MM')

model_vqvae = vqvae.VQVAE(config_vq_vae,writer)
model_vqvae.load_checkpoint(vqvae_path + '/vqvae_UKBB_best.pt')
model_vqvae.model.eval()

for loop_repeat in range(repeat):
    print(f"Loop {loop_repeat + 1}")
    print("Encoding with VQVAE")
    print(f"data.shape: {data.shape}")
    dataset_test = CustomDataset(data.astype(np.float32), config_vq_vae.device)
    test_loader = DataLoader(dataset_test, shuffle=True, batch_size=config_vq_vae.training.batch_size)
    encoded_data_all = model_vqvae.encode_without_codebook(test_loader)
    print(f"encoded_data.shape: {encoded_data_all.shape}")
    n_channels = encoded_data_all.shape[1]
    print(f"n_channels: {n_channels}")
    all_c_cf = np.zeros((nsamples,n_channels,ncomps))
    datashape1 = ncauses + (nevecs * n_channels)
    #for channel in range(n_channels):
        #with open(scalers_path + f'/c{channel}_scalers.pkl','rb') as f:  
        #    scalers = pickle.load(f)
    for ev in range(0,ncomps-nbasecomps,nevecs-nbasecomps):
    #for ev in range(0,all_data_dim[order_slice],nevecs):
        #encoded_obs =  encoded_data_all[:,:,ev:ev+nevecs].reshape(encoded_data_all.shape[0],-1)
        encoded_obs = encoded_data_all[:,:,ev:ev+nevecs]
        encoded_obs = np.swapaxes(encoded_obs,1,2)
        encoded_obs = encoded_obs.reshape(encoded_obs.shape[0],-1)
        try:
            macaw = MACAW.MACAW(config)
            macaw.load_model(model_path + f'/macaw_ukbb_PCA3D_{ev}.pt',
                            edges,priors,datashape1)
            '''
            macaw = torch.load(model_path + f'/c{channel}_macaw_ukbb_PCA3D_{ev}.pt', map_location=torch.device(config.device))
            macaw.model.to(config.device)
            for each_flow in macaw.flow_list:
                each_flow.to(config.device)
                each_flow.device = config.device
            macaw.device = config.device
            scaler = scalers[f"{ev}"]
            encoded_obs = scaler.transform(encoded_obs)
            max_value = encoded_obs.max()
            min_value = encoded_obs.min()
            '''
            #X_obs = np.hstack([c_obs,encoded_obs])  
            macaw.model.eval()
            with torch.no_grad():
                # abduction:
                X_obs = np.hstack([np.array(sex_obs)[:,np.newaxis], np.array(age_obs)[:,np.newaxis], encoded_obs])
                z_obs = macaw._forward_flow(X_obs) 
                # action (get latent variable value under counterfactual)
                #x_cf = np.copy(X_obs)
                
                #x_cf[:, 1] = cf_age_obs
                #z_cf_val = macaw._forward_flow(x_cf)
                
                # prediction (pass through the flow):
                #z_obs[:, 1] = z_cf_val[:, 1]
                cc = macaw._backward_flow(z_obs)
                cc_nan_places = np.argwhere(np.isnan(cc))
                if (len(cc_nan_places) > 0):
                    print(f"nan values produced on cc channel evec {ev}:")
                    print(cc_nan_places)
                    cc = np.nan_to_num(cc)
            #d_encoded_cf = cc[:,ncauses:]
            # limiting problems with infinitive values
            #d_encoded_cf[d_encoded_cf>max_value] = max_value
            #d_encoded_cf[d_encoded_cf<min_value] = min_value
            #c_cf = cf[:,:ncauses]
            #cc[:,ncauses:] = scaler.inverse_transform(cc[:,ncauses:])
            #cc[:,ncauses:] = scaler.inverse_transform(d_encoded_cf)
        except FileNotFoundError as e:
            print(e)
            print(f"File not finded: macaw_ukbb_PCA3D_{ev}.pt")
            cc[:,ncauses:] = encoded_obs.copy()
            #d_encoded_cf = encoded_obs
        except ValueError as value_error:
            #print(f"IDs of subjects: {random_eids}")
            print(cc.shape)
            print(np.isinf(cc).nonzero())
            print(f"Error in model macaw_ukbb_PCA3D_{ev}.pt")
            raise Exception('Just to stop and see', value_error)
        '''
        '''
        cc_reshaped = cc[:,ncauses:].reshape(encoded_data_all.shape[0],nevecs,n_channels)
        cc_reshaped = np.swapaxes(cc_reshaped,1,2)
        all_c_cf[:,:,ev:ev+nevecs] = cc_reshaped
        #d_encodes.append(d_encoded_cf)

            
    #age_obs = age_obs + min_age
    #sex = ['Male' if round(s) else 'Female' for s in age_obs]
    #titles = [f'Age:{a}, Sex:{s}' for a,s in zip(age_obs,sex)]

    print(f"Shape do all_c_cf: {all_c_cf.shape}")

    nan_places = np.argwhere(np.isnan(all_c_cf))
    print("Positions of nan values")
    print(nan_places)
    print("")

    sex_cf = ['Male' if round(s) else 'Female' for s in cc[:,0]]
    age_cf = cf_age_obs + min_age
    titles_cf = [f'Age:{np.round(a)}, Sex:{s}' for a,s in zip(age_cf,sex_cf)]
            
    all_c_cf = all_c_cf.reshape((all_c_cf.shape[0],all_c_cf.shape[1],17,25,25))
    all_decoded_cf = model_vqvae.decode_with_codebook(all_c_cf.astype(np.float32)) #+ residuals

    data = all_decoded_cf
    data = data[:, None, : ,: ,:]
    print(f"Shape do all_decoded_cf: {all_decoded_cf.shape}")

import nibabel as nib

#all_decoded_cf_2 = np.array(all_decoded_cf)


def count_files_folder(path_file_to_count):
    return len([name for name in os.listdir(path_file_to_count) if os.path.isfile(os.path.join(path_file_to_count, name))])


number_files_output = count_files_folder(output_image)

for individual in range(nsamples):
    #numpy_image = np.array([])
    numpy_image = all_decoded_cf[individual]
    #numpy_image = np.swapaxes(numpy_image,0,1)
    # Getting the orifinal affine and header
    numpy_image = np.swapaxes(numpy_image,0,1)
    numpy_image = np.swapaxes(numpy_image,1,2)
    image_path = original_images + '/test/' + str(random_eids[individual]) + '.nii.gz'
    original_image = nib.load(image_path)
    # Saving new generated images
    ni_img = nib.Nifti1Image(numpy_image, original_image.affine, original_image.header)
    nib.save(ni_img, output_image + "/" + str(random_eids[individual])  + f"_{int(age_cf[individual])}_{sex_cf[individual]}.nii.gz")
    if count_files_folder(output_image) > number_files_output:
        number_files_output = count_files_folder(output_image)
    else:
        print(f"File {random_eids[individual]} not saved for some reason")
        print(numpy_image.shape)
        print(np.argwhere(np.isnan(numpy_image)))
        break


print(f"Number of unique ID files is: {count_files_folder(output_image)}")