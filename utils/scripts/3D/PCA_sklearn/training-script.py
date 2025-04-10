#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import numpy as np
import argparse

from  torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, CenterCrop
from torchvision import transforms 
from sklearn.preprocessing import StandardScaler

import torch
import torch.distributions as td
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import yaml

from pathlib import Path
sys.path.append('/home/erik.ohara/macaw/')
from utils.datasets import UKBBT1Dataset
from utils.customTransforms import ToFloatUKBB
from utils.helpers import dict2namespace
from macaw import MACAW

import pickle
import pandas as pd

# getting the slice from command line
parser = argparse.ArgumentParser()
parser.add_argument('z_slice')
args = parser.parse_args()
z_slice = int(args.z_slice)

nevecs = 50
ncauses = 2
ncomps = 1500
nbasecomps = 25
rs = 150
ukbb_path = '/home/erik.ohara/UKBB'
ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/train'
pca_path = '/work/forkert_lab/erik/PCA/slices-z-sklearn'
macaw_path = '/home/erik.ohara/macaw'
output_path = '/work/forkert_lab/erik/MACAW/models'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'

#dataset = UKBBT1Dataset(ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), CenterCrop(rs)]))
#all_loader = DataLoader(dataset, batch_size=batch_size)



#data = np.concatenate([d.numpy() for d in all_loader],axis=0)
#data = data.reshape(data.shape[0],-1)
data = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(z_slice)) 
print("Data loaded")
#print("Data reshaped")

with open(pca_path + '/evecs_slice_{}.pkl'.format(z_slice),'rb') as f:  
    evecs = pickle.load(f)

print("PCA loaded")

def encode(data, evecs):
    return np.matmul(data,evecs.T)

def decode(data,evecs):
    return np.matmul(data,evecs)

data_path = ukbb_path + '/ukbb_img.csv'

df = pd.read_csv(data_path,low_memory=False)
print(f"The original size of the dataframe is {df.shape}")
# adding noise to sex and normalizing age
#df['Sex'] = df['Sex'] + np.random.rand(df.shape[0]) - 0.5
#scaler_a = StandardScaler()
#df['Age'] = scaler_a.fit_transform(df[['Age']])

# getting only the training set
df_train = pd.DataFrame(columns=df.columns)
for each_file in os.listdir(ukbb_path_T1_slices):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        df_train = pd.concat([df[df['eid'] == int(file_id)],df_train.loc[:]]).reset_index(drop=True)
df_train.sort_values(by=['eid'], inplace=True)

df = df_train
print(f"The size of the dataframe (just train set) is {df.shape}")
causes = df[['Age','Sex']].to_numpy()

sex = df['Sex'] 
age = df['Age'] - df['Age'].min()

# causal Graph
sex_to_latents = [(0,i) for i in range(ncauses,nevecs+ncauses)]
age_to_latents = [(1,i) for i in range(ncauses,nevecs+ncauses)]
autoregressive_latents = [(i,j) for i in range(ncauses,nevecs+ncauses-1) for j in range(i+1,nevecs+ncauses)]
edges = sex_to_latents + age_to_latents + autoregressive_latents

with open(macaw_path + '/config/ukbb.yaml', 'r') as f:
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

# In[ ]:

print("Start training")

loss_vals_all= []
for e in range(0,ncomps-nbasecomps,nevecs-nbasecomps):
#for e in range(0,ncomps,nevecs):
    folder_path = output_path + f'/PCA_sklearn_2/slice_{z_slice}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = folder_path + f'/macaw_ukbb_PCA_{e}.pt'

    encoded_data =  encode(data,evecs[e:e+nevecs])
    #scaler = StandardScaler()
    #encoded_data = scaler.fit_transform(encoded_data)
    print(e)
    
    if not os.path.exists(save_path):    
        #X = np.hstack([causes, encoded_data])    
        X = np.hstack([sex[:,np.newaxis], age[:,np.newaxis], encoded_data]) 

        macaw = MACAW.MACAW(config)
        #loss_vals = macaw.fit(X,edges, augment=True)
        loss_vals = macaw.fit_with_priors(X,edges, priors)
        df_loss_vals = pd.DataFrame(loss_vals)
        if (df_loss_vals.isnull().values.any()):
            print("Tem um nulo no {}".format(e))
            break
        #elif (loss_vals[1][-1] > 0):
        #    print("Terminou com valor positivo nos evec {}".format(e))
        #    break
        loss_vals_all.append(loss_vals)

        #macaw.save_best_model()
        torch.save(macaw,save_path)
