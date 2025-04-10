#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
import torch.distributions as td
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import yaml

sys.path.append('/home/erik.ohara/macaw/')
from utils.helpers import dict2namespace
from macaw import MACAW

import pickle
import pandas as pd

nevecs = 50
ncauses = 2
init_comp = 0
ncomps = 15000
nbasecomps = 25
ukbb_path = '/home/erik.ohara/UKBB'
pca_path = '/work/forkert_lab/erik/PCA3D_full'
macaw_path = '/home/erik.ohara/macaw'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/full_3D'
ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/train'
scalers_path = '/work/forkert_lab/erik/MACAW/scalers/PCA3D_15000_full'
output_path = f"/work/forkert_lab/erik/MACAW/models/PCA3D_15000_full"

data = np.load(reshaped_path + '/reshaped_3D_train.npy')
data = data.reshape(data.shape[0],-1)
print("Data train loaded")

data_val = np.load(reshaped_path + '/reshaped_3D_val.npy')
data_val = data_val.reshape(data_val.shape[0],-1)
print("Data val loaded")

with open(pca_path + "/evecs.pkl",'rb') as f:  
    evecs3D = pickle.load(f)
print("PCA loaded")

def encode(data, evecs):
    return np.matmul(data,evecs.T)

def decode(data,evecs):
    return np.matmul(data,evecs)

# Getting Age and Sex data
data_path = ukbb_path + '/ukbb_img.csv'
df = pd.read_csv(data_path,low_memory=False)
min_age = df['Age'].min()

sex = df['Sex'] 
age = df['Age'] - min_age

# Loading configurations
with open(macaw_path + '/config/ukbbHPCA.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Priors
P_sex = np.sum(sex)/len(sex)

unique_values, counts = np.unique(age, return_counts=True)
P_age = np.array(counts)/np.sum(counts)
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

print("Start training")
#loss_vals_all= {}
scalers = {}
for e in range(init_comp,ncomps-nbasecomps,nevecs-nbasecomps):
#for e in range(0,ncomps,nevecs):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = output_path + f'/macaw_ukbb_PCA3D_{e}.pt'

    encoded_data = encode(data,evecs3D[e:e+nevecs])
    encoded_data_val = encode(data_val,evecs3D[e:e+nevecs])
    scaler = StandardScaler()
    encoded_data = scaler.fit_transform(encoded_data)
    encoded_data_val = scaler.transform(encoded_data_val)
    scalers[f"{e}"] = scaler
    print(e)
    
    if not os.path.exists(save_path):    
        #X = np.hstack([causes, encoded_data])  
        X = np.hstack([np.array(sex)[:,np.newaxis], np.array(age)[:,np.newaxis], encoded_data])   
        X_val = np.hstack([np.array(sex_val)[:,np.newaxis], np.array(age_val)[:,np.newaxis], encoded_data_val])   

        macaw = MACAW.MACAW(config)
        #loss_vals = macaw.fit(X,edges, augment=True)
        loss_vals = macaw.fit_with_priors(X,edges, priors, validation=X_val,save_path=save_path,patience=50)
        df_loss_vals = pd.DataFrame(loss_vals)
        if (df_loss_vals.isnull().values.any()):
            print("Tem um nulo no {}".format(e))
            break
        #loss_vals_all[f"{e}"] = loss_vals
        
        #macaw.save_best_model()
        #torch.save(macaw,save_path)
        '''
        with open(output_path + f'/losses.pkl', 'wb') as lossesfile:
            pickle.dump(loss_vals_all, lossesfile)
        '''

with open(scalers_path + f'/scalersPCA3D.pkl', 'wb') as filefinal:
    pickle.dump(scalers, filefinal)



