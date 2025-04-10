#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
import torch.distributions as td
from tensorboardX import SummaryWriter
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import yaml

sys.path.append('/home/erik.ohara/macaw/')
from utils.helpers import dict2namespace
from compression.vqvae import vqvae
from macaw import MACAW
from  torch.utils.data import DataLoader
from utils.datasets import CustomDataset

import pickle
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('encoded_dim')
parser.add_argument('z_slice')
args = parser.parse_args()
#encoded_dim = int(args.encoded_dim)
z_slice = args.z_slice

nevecs = 50
ncauses = 2
ncomps = 625
nbasecomps = 25
ukbb_path = '/home/erik.ohara/UKBB'
macaw_path = '/home/erik.ohara/macaw'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'
output_path = f"/work/forkert_lab/erik/MACAW/models/macaw_VQVAE2D/slice_{z_slice}"
vqvae_path = f'/work/forkert_lab/erik/MACAW/models/vqvae2D_8/slice_{z_slice}'

data = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(z_slice)) 
data = data.reshape(data.shape[0],1,150,150)
#data = data.reshape(data.shape[0],-1)
print(f"Data train loaded: {data.shape}")

data_val = np.load(reshaped_path + '/reshaped_val_slice_{}.npy'.format(z_slice)) 
data_val = data_val.reshape(data_val.shape[0],1,150,150)
#data_val = data_val.reshape(data_val.shape[0],-1)
print(f"Data val loaded: {data_val.shape}")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Getting Age and Sex data
data_path = ukbb_path + '/ukbb_img.csv'
df = pd.read_csv(data_path,low_memory=False)
min_age = df['Age'].min()

sex = df['Sex'] 
age = df['Age'] - min_age

# Loading configurations
with open(macaw_path + '/config/ukbbVQVAE.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Priors
P_sex = np.sum(sex)/len(sex)

unique_values, counts = np.unique(age, return_counts=True)
P_age = counts/np.sum(counts)
priors = [(slice(0,1),td.Bernoulli(torch.tensor([P_sex]).to(config.device))), # sex
          (slice(1,2),td.Categorical(torch.tensor([P_age]).to(config.device))), # age
          (slice(ncauses,nbasecomps+ncauses),td.Normal(torch.zeros(nbasecomps).to(config.device), torch.ones(nbasecomps).to(config.device))), # base_comps
          (slice(nbasecomps+ncauses,nevecs+ncauses),td.Normal(torch.zeros(nevecs-nbasecomps).to(config.device), torch.ones(nevecs-nbasecomps).to(config.device))), # new_comps
         ]
#df['Sex'] = df['Sex'] + np.random.rand(df.shape[0]) - 0.5
#scaler_a = StandardScaler()
#df['Age'] = scaler_a.fit_transform(df[['Age']])
print(f"The original size of the dataframe is {df.shape}")

# getting only the training set
df_train = pd.read_csv(ukbb_path + '/train.csv',low_memory=False)
df_val = pd.read_csv(ukbb_path + '/val.csv',low_memory=False)
df_train.sort_values(by=['eid'], inplace=True)
df_val.sort_values(by=['eid'], inplace=True)
'''
df_train = pd.DataFrame(columns=df.columns)
for each_file in os.listdir(ukbb_path_T1_slices):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        df_train = pd.concat([df[df['eid'] == int(file_id)],df_train.loc[:]]).reset_index(drop=True)
df_train.sort_values(by=['eid'], inplace=True)
'''

df = df_train
print(f"The size of the dataframe (just train set) is {df.shape}")

# Saving the Causal relationships
#causes = df[['Age','Sex']].to_numpy()

sex = df['Sex'] 
sex_val = df_val['Sex'] 
age = df['Age'] - min_age
age_val = df_val['Age'] - min_age

# causal Graph
sex_to_latents = [(0,i) for i in range(ncauses,nevecs+ncauses)]
age_to_latents = [(1,i) for i in range(ncauses,nevecs+ncauses)]
autoregressive_latents = [(i,j) for i in range(ncauses,nevecs+ncauses-1) for j in range(i+1,nevecs+ncauses)]
edges = sex_to_latents + age_to_latents + autoregressive_latents


with open(macaw_path +'/compression/vqvae/vqvae2D.yaml', 'r') as f:
    config_raw_vq_vae = yaml.load(f, Loader=yaml.FullLoader)
config_vq_vae = dict2namespace(config_raw_vq_vae)
config_vq_vae.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"CPU or cuda vqvae: {config_vq_vae.device}")


dataset_train = CustomDataset(data.astype(np.float32), config_vq_vae.device)
dataset_val = CustomDataset(data_val.astype(np.float32), config_vq_vae.device)

train_loader = DataLoader(dataset_train, shuffle=False, batch_size=config_vq_vae.training.batch_size)
val_loader = DataLoader(dataset_val, shuffle=False, batch_size=config_vq_vae.training.batch_size)


print("Start enconding the data")
print(f"CPU or cuda vqvae: {config_vq_vae.device}")

writer = SummaryWriter(macaw_path + 'logs/vqvae_MM')

model_vqvae = vqvae.VQVAE(config_vq_vae,writer)
model_vqvae.load_checkpoint(vqvae_path + '/vqvae_UKBB_best.pt')



#encoded_data, eids = ae.encode(test_loader)

encoded_data_all = model_vqvae.encode_without_codebook(train_loader)
encoded_data_val_all = model_vqvae.encode_without_codebook(val_loader)
print(f"encoded_data.shape: {encoded_data_all.shape}")
print(f"encoded_data_val_all.shape: {encoded_data_val_all.shape}")

print("Start training")
loss_vals_all= {}
#scalers = {}
for e in range(0,ncomps-nbasecomps,nevecs-nbasecomps):
#for e in range(0,ncomps,nevecs):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = output_path + f'/macaw_ukbb_PCA3D_{e}.pt'

    #encoded_data = encoded_data_all[:,e:e+nevecs]
    #encoded_data_val = encoded_data_val_all[:,e:e+nevecs]
    encoded_data = encoded_data_all[:,:,e:e+nevecs]
    encoded_data = np.swapaxes(encoded_data,1,2)
    encoded_data = encoded_data.reshape(encoded_data.shape[0],-1)
    encoded_data_val = encoded_data_val_all[:,:,e:e+nevecs]
    encoded_data_val = np.swapaxes(encoded_data_val,1,2)
    encoded_data_val = encoded_data_val.reshape(encoded_data_val.shape[0],-1)
    print(e)
    
    if not os.path.exists(save_path):    
        #X = np.hstack([causes, encoded_data])  
        X = np.hstack([np.array(sex)[:,np.newaxis], np.array(age)[:,np.newaxis], encoded_data])   
        X_val = np.hstack([np.array(sex_val)[:,np.newaxis], np.array(age_val)[:,np.newaxis], encoded_data_val])   

        macaw = MACAW.MACAW(config)
        #loss_vals = macaw.fit(X,edges, augment=True)
        loss_vals = macaw.fit_with_priors(X,edges, priors, validation=X_val, patience=50,save_path=save_path)
        df_loss_vals = pd.DataFrame(loss_vals)
        if (df_loss_vals.isnull().values.any()):
            print("Tem um nulo no {}".format(e))
            break
        loss_vals_all[f"{e}"] = loss_vals
        
        #macaw.save_best_model()
        #torch.save(macaw,save_path)

        with open(output_path + f'/loss_vals_all.pkl', 'wb') as lossesfile:
            pickle.dump(loss_vals_all, lossesfile)

#with open(scalers_path + f'/scalersPCA3D.pkl', 'wb') as filefinal:
#    pickle.dump(scalers, filefinal)



