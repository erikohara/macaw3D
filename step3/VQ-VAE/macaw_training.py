#!/usr/bin/env python
# coding: utf-8

import sys
import wandb
import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
from  torch.utils.data import DataLoader

import torch
import torch.distributions as td
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import yaml

sys.path.append('/home/erik.ohara/macaw/')
from utils.helpers import dict2namespace
from compression.vqvae import vqvae
from macaw import MACAW
from utils.datasets import CustomDataset

import pickle
import pandas as pd
import argparse

# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('ncomps_init')
parser.add_argument('ncomps')
args = parser.parse_args()
ncomps_init = int(args.ncomps_init)
ncomps = int(args.ncomps)


nchannels = 8
#channel_ini = 0
nevecs = 50
ncauses = 2
#ncomps_init = 0
#ncomps = 10625
#ncomps = 3525
nbasecomps = 25
macaw_patience = 50
standardscaler = False
ukbb_path = '/home/erik.ohara/UKBB'
pca_path = '/work/forkert_lab/erik/PCA3D'
macaw_path = '/home/erik.ohara/macaw'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/train'
scalers_path = '/work/forkert_lab/erik/MACAW/scalers/macaw_vqvae8_50nevecs_nan_to_num_scaler'
output_path = f"/work/forkert_lab/erik/MACAW/models/macaw_vqvae8_50_nevecs"
vqvae_path = '/work/forkert_lab/erik/MACAW/models/vqvae3D_8'

data_train = np.load(reshaped_path + '/reshaped_3D_102_150_150_train.npy')
print("Data train loaded")

data_val = np.load(reshaped_path + '/reshaped_3D_102_150_150_val.npy')
print("Data val loaded")

print("Starting code")
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
print(f"CPU or cuda macaw: {config.device}")
print(config)

# Priors
P_sex = np.sum(sex)/len(sex)

unique_values, counts = np.unique(age, return_counts=True)
P_age = counts/np.sum(counts)
priors = [(slice(0,1),td.Bernoulli(torch.tensor([P_sex]).to(config.device))), # sex
          (slice(1,2),td.Categorical(torch.tensor([P_age]).to(config.device))), # age
          (slice(ncauses,(nbasecomps*nchannels)+ncauses),td.Normal(torch.zeros(nbasecomps*nchannels).to(config.device), torch.ones(nbasecomps*nchannels).to(config.device))), # base_comps
          (slice((nbasecomps*nchannels)+ncauses,(nevecs*nchannels)+ncauses),td.Normal(torch.zeros((nevecs-nbasecomps)*nchannels).to(config.device), torch.ones((nevecs-nbasecomps)*nchannels).to(config.device))), # new_comps
         ]
#df['Sex'] = df['Sex'] + np.random.rand(df.shape[0]) - 0.5
#scaler_a = StandardScaler()
#df['Age'] = scaler_a.fit_transform(df[['Age']])
print(f"The original size of the dataframe is {df.shape}")

# getting only the training set
df_train = pd.read_csv(ukbb_path + '/train.csv',low_memory=False)
df_val = pd.read_csv(ukbb_path + '/val.csv',low_memory=False)
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
sex_to_latents = [(0,i) for i in range(ncauses,nevecs*nchannels+ncauses)]
age_to_latents = [(1,i) for i in range(ncauses,nevecs*nchannels+ncauses)]
autoregressive_latents = [(i,j) for i in range(ncauses,nevecs*nchannels+ncauses-1) for j in range(i+1,nevecs*nchannels+ncauses)]
edges = sex_to_latents + age_to_latents + autoregressive_latents

# loading VQVAE
with open(macaw_path +'/compression/vqvae/vqvae.yaml', 'r') as f:
    config_raw_vq_vae = yaml.load(f, Loader=yaml.FullLoader)
config_vq_vae = dict2namespace(config_raw_vq_vae)
config_vq_vae.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"CPU or cuda vqvae: {config_vq_vae.device}")

writer = SummaryWriter(macaw_path + 'logs/vqvae_MM')

model_vqvae = vqvae.VQVAE(config_vq_vae,writer)
model_vqvae.load_checkpoint(vqvae_path + '/vqvae_UKBB_best.pt')

if not os.path.exists(vqvae_path + '/encoded_data_all.pkl'): 
    raise Exception('Nao achou o encoded value com vqvae')
    print("Encoding with VQVAE")
    dataset_train = CustomDataset(data_train.astype(np.float32), config_vq_vae.device)
    dataset_val = CustomDataset(data_val.astype(np.float32), config_vq_vae.device)
    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=config_vq_vae.training.batch_size)
    val_loader = DataLoader(dataset_val, shuffle=True, batch_size=config_vq_vae.training.batch_size)


    encoded_data_all = model_vqvae.encode_without_codebook(train_loader)
    encoded_data_val_all = model_vqvae.encode_without_codebook(val_loader)
    print(f"encoded_data.shape: {encoded_data_all.shape}")
    print(f"encoded_data_val_all.shape: {encoded_data_val_all.shape}")
    with open(vqvae_path + '/encoded_data_all.pkl', 'wb') as file_train_data:
                pickle.dump(encoded_data_all, file_train_data)
    with open(vqvae_path + '/encoded_data_val_all.pkl', 'wb') as file_val_data:
                pickle.dump(encoded_data_val_all, file_val_data)
else:
    with open(vqvae_path + '/encoded_data_all.pkl','rb') as file_train_data:  
        encoded_data_all = pickle.load(file_train_data)
    with open(vqvae_path + '/encoded_data_val_all.pkl','rb') as file_val_data:  
        encoded_data_val_all = pickle.load(file_val_data)

wandb_config = {
            "macaw_config": config,
            "vqvae_config": config_vq_vae,
            'nevecs': nevecs,
            'ncomps': ncomps,
            'nbasecomps': nbasecomps,
            "ncauses": ncauses,
            'output_path': output_path,
            'standardscaler': standardscaler,
            'macaw_patience': macaw_patience,
            'slurm_job_id': os.environ['SLURM_JOB_ID']
        }

print("Start training")
loss_vals_all= {}
if standardscaler:
    scalers = {}
#for channel in range(channel_ini, channel_ini + nchannels):
#loss_vals_all[f"{channel}"] = {}
#wandb_run_id = 'js1l3r0t'
with wandb.init(project=f"VQ-VAE_macaw_training_{nevecs}nevecs_final", config=wandb_config): #, id=wandb_run_id, resume="must"):
    for e in range(ncomps_init,ncomps-nbasecomps,nevecs-nbasecomps):
    #for e in range(0,ncomps,nevecs):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = output_path + f'/macaw_ukbb_PCA3D_{e}.pt'
        encoded_data = encoded_data_all[:,:,e:e+nevecs]
        encoded_data = np.swapaxes(encoded_data,1,2)
        encoded_data = encoded_data.reshape(encoded_data.shape[0],-1)
        encoded_data_val = encoded_data_val_all[:,:,e:e+nevecs]
        encoded_data_val = np.swapaxes(encoded_data_val,1,2)
        encoded_data_val = encoded_data_val.reshape(encoded_data_val.shape[0],-1)
        if standardscaler:
            scaler = StandardScaler()
            encoded_data = scaler.fit_transform(encoded_data)
            encoded_data_val = scaler.transform(encoded_data_val)
            scalers[f"{e}"] = scaler
        print(f"evec: {e}")
        
        if not os.path.exists(save_path):    
            #X = np.hstack([causes, encoded_data])  
            X = np.hstack([np.array(sex)[:,np.newaxis], np.array(age)[:,np.newaxis], encoded_data])   
            X_val = np.hstack([np.array(sex_val)[:,np.newaxis], np.array(age_val)[:,np.newaxis], encoded_data_val])   

            macaw = MACAW.MACAW(config)
            #loss_vals = macaw.fit(X,edges, augment=True)
            loss_vals = macaw.fit_with_priors(X,edges, priors, 
                                                validation=X_val,
                                                save_path=save_path,
                                                patience=macaw_patience)
            
            for step in range(len(loss_vals[0])):
                wandb.log({ 
                    f'e{e}_train_loss': loss_vals[0][step],
                    f'e{e}_val_loss': loss_vals[1][step]
                })
            df_loss_vals = pd.DataFrame(loss_vals)
            if (df_loss_vals.isnull().values.any()):
                print("Tem um nulo no {}".format(e))
                break
            loss_vals_all[f"{e}"] = (loss_vals)

            with open(output_path + f'/losses_{ncomps_init}.pkl', 'wb') as file:
                pickle.dump(loss_vals_all, file)
            
            #macaw.save_best_model()
            #torch.save(macaw,save_path)
    if standardscaler:
        with open(scalers_path + f'/scalers.pkl', 'wb') as filefinal:
            pickle.dump(scalers, filefinal)