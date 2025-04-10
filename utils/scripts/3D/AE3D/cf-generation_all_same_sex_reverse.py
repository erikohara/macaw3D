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


nevecs = 50
ncauses = 2
ncomps = 6000
nbasecomps = 25
cf_sex = 0
ukbb_path = '/home/erik.ohara/UKBB'
model_path = f"/work/forkert_lab/erik/MACAW/models/macaw_AE3D_6000"
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
output_image = f'/work/forkert_lab/erik/MACAW/cf_images/macaw_AE3D_6000_sex_{cf_sex}_reverse'
cf_age_path = '/work/forkert_lab/erik/MACAW/2_5'
original_images = "/work/forkert_lab/erik/T1_warped"
ae_path = '/work/forkert_lab/erik/MACAW/models/AE3D_6000_previous'
number_images = 2370
batch_size_ae = 2

# In[13]:
print("Starting code")
if not os.path.exists(output_image):
    os.makedirs(output_image)

data = np.load(reshaped_path + '/reshaped_3D_102_150_150_macaw_AE3D_6000_sex_0.npy')
#data = data.reshape(data.shape[0],-1)
print(f"Data loaded. data.shape: {data.shape}")

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
          (slice(ncauses,nbasecomps+ncauses),td.Normal(torch.zeros(nbasecomps).to(config.device), torch.ones(nbasecomps).to(config.device))), # base_comps
          (slice(nbasecomps+ncauses,nevecs+ncauses),td.Normal(torch.zeros(nevecs-nbasecomps).to(config.device), torch.ones(nevecs-nbasecomps).to(config.device))), # new_comps
         ]

# causal Graph
sex_to_latents = [(0,i) for i in range(ncauses,nevecs+ncauses)]
age_to_latents = [(1,i) for i in range(ncauses,nevecs+ncauses)]
autoregressive_latents = [(i,j) for i in range(ncauses,nevecs+ncauses-1) for j in range(i+1,nevecs+ncauses)]
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
cf_sex_obs = np.full((len(sex_obs)), cf_sex)
random_eids = all_eid[:,0]

# create list with +5 and -5 for the countefactual age
# create list with +5 and -5 for the countefactual age
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
#cf_vals = {1:cf_age_definition - min_age}
#residuals = data - decode(encode(data,evecs3D[:ncomps]),evecs3D[:ncomps])
#all_d_encodes = []
# loading VQVAE




print("Encoding with AE")
dataset_test = CustomDataset(data.astype(np.float32), config.device)
test_loader = DataLoader(dataset_test, batch_size=batch_size_ae)

ae = torch.load(ae_path + '/ae_UKBB_best.pt',map_location=torch.device(config.device))
ae.device = config.device

encoded_data_all = ae.encode(test_loader)
print(f"encoded_data.shape: {encoded_data_all.shape}")

all_c_cf = np.zeros((nsamples,ncomps))
datashape1 = ncauses + nevecs

for ev in range(0,ncomps-nbasecomps,nevecs-nbasecomps):
#for ev in range(0,all_data_dim[order_slice],nevecs):
    encoded_obs =  encoded_data_all[:,ev:ev+nevecs] 
    try:
        macaw = MACAW.MACAW(config)
        #macaw.load_model(model_path + f'/macaw_ukbb_PCA3D_{ev}.pt',edges,priors,datashape1)
        macaw = torch.load(model_path + f'/macaw_ukbb_PCA3D_{ev}.pt', map_location=torch.device(config.device))
        '''
        macaw = torch.load(model_path + f'/c{channel}_macaw_ukbb_PCA3D_{ev}.pt', map_location=torch.device(config.device))
        macaw.model.to(config.device)
        for each_flow in macaw.flow_list:
            each_flow.to(config.device)
            each_flow.device = config.device
        macaw.device = config.device
        '''
        #scaler = scalers[f"{ev}"]
        #encoded_obs = scaler.transform(encoded_obs)
        #max_value = encoded_obs.max()
        #min_value = encoded_obs.min()
        #X_obs = np.hstack([c_obs,encoded_obs])  
        macaw.model.eval()
        with torch.no_grad():
            # abduction:
            X_obs = np.hstack([np.array(cf_sex_obs)[:,np.newaxis], np.array(age_obs)[:,np.newaxis], encoded_obs])
            #cc = macaw.counterfactual(X_obs,cf_vals)
            z_obs = macaw._forward_flow(X_obs) 
            # action (get latent variable value under counterfactual)
            x_cf = np.copy(X_obs)
            
            x_cf[:, 0] = sex_obs
            z_cf_val = macaw._forward_flow(x_cf)
            
            # prediction (pass through the flow):
            z_obs[:, 0] = z_cf_val[:, 0]
            cc = macaw._backward_flow(z_obs)
            '''
            '''
        #d_encoded_cf = cf[:,ncauses:]
        # limiting problems with infinitive values
        #d_encoded_cf[d_encoded_cf>max_value] = max_value
        #d_encoded_cf[d_encoded_cf<min_value] = min_value
        #c_cf = cf[:,:ncauses]
        #cc[:,ncauses:] = scaler.inverse_transform(cc[:,ncauses:])
    except FileNotFoundError as e:
        print(e)
        print(f"File not finded: macaw_ukbb_PCA3D_{ev}.pt")
        cc[:,ncauses:] = encoded_obs.copy()
        #d_encoded_cf = encoded_obs
    except ValueError as value_error:
        print(f"IDs of subjects: {random_eids}")
        print(cc.shape)
        print(np.isinf(cc).nonzero())
        print(f"Error in model macaw_ukbb_PCA3D_{ev}.pt")
        raise Exception('Just to stop and see', value_error)
    '''
    '''
    all_c_cf[:,ev:ev+nevecs] = cc[:,ncauses:]
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
age_cf = age_obs + min_age
#titles_cf = [f'Age:{np.round(a)}, Sex:{s}' for a,s in zip(age_cf,sex_cf)]
        
#all_c_cf = all_c_cf.reshape((all_c_cf.shape[0],17,25,25))

imgs = DataLoader(CustomDataset(all_c_cf.astype(np.float32)), batch_size=batch_size_ae)
all_decoded_cf = ae.decode(imgs)

import nibabel as nib

#all_decoded_cf_2 = np.array(all_decoded_cf)
print(f"Shape do all_decoded_cf: {all_decoded_cf.shape}")

def count_files_folder(path_file_to_count):
    return len([name for name in os.listdir(path_file_to_count) if os.path.isfile(os.path.join(path_file_to_count, name))])


number_files_output = count_files_folder(output_image)

for individual in range(nsamples):
    numpy_image = np.array([])
    numpy_image = all_decoded_cf[individual].reshape(102,150,150)
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