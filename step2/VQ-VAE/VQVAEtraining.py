#!/usr/bin/env python
# coding: utf-8

import sys
import torch
import pickle
import yaml
import pandas as pd
import numpy as np

from  torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 
from tensorboardX import SummaryWriter

import os
import wandb
os.environ['TF_DETERMINISTIC_OPS'] = '1'

sys.path.append('/home/erik.ohara/macaw/')
from utils.helpers import dict2namespace 
from compression.vqvae import vqvae
from utils.datasets import UKBBT13DDataset, CustomDataset
from utils.customTransforms import ToFloatUKBB, Crop3D

batch_size = 16
crop_size = (102,150,150)
#encoded_dim = 15000
ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/train'
macaw_path = '/home/erik.ohara/macaw/'
ukbb_path_T1_val = '/work/forkert_lab/erik/T1_warped/val'
df_ukbb_train = '/home/erik.ohara/UKBB/train.csv'
df_ukbb_val = '/home/erik.ohara/UKBB/val.csv'
output_path = '/work/forkert_lab/erik/MACAW/models/vqvae3D_4'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
load_checkpoint = False  


# In[12]:
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_train = np.load(reshaped_path + '/reshaped_3D_102_150_150_train.npy')
print("Data train loaded")

data_val = np.load(reshaped_path + '/reshaped_3D_102_150_150_val.npy')
print("Data val loaded")

dataset_train = CustomDataset(data_train.astype(np.float32), device)
dataset_val = CustomDataset(data_val.astype(np.float32), device)

train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)

'''
df_train = pd.read_csv(df_ukbb_train,low_memory=False)
df_val = pd.read_csv(df_ukbb_val,low_memory=False)

dataset_train = UKBBT13DDataset(df_train,ukbb_path_T1_slices, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
train_loader = DataLoader(dataset_train, batch_size=batch_size)

dataset_val = UKBBT13DDataset(df_val,ukbb_path_T1_val, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
valid_loader = DataLoader(dataset_val, batch_size=batch_size)
'''

with open(macaw_path +'/compression/vqvae/vqvae.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
    
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_epochs = config.training.epochs

writer = SummaryWriter(macaw_path + 'logs/vqvae_MM')

with wandb.init(project="VQ-VAE_training", config=config):

    model = vqvae.VQVAE(config,writer)
    losses = {'train_loss':[],'val_loss':[]}
    best_val_loss = 99999999

    initial_epoch = 0

    if load_checkpoint:
        initial_epoch, losses, best_val_loss = model.load_checkpoint(output_path + '/checkpoint.pt')


    print(f"Start training from epoch {initial_epoch} with best loss at {best_val_loss}")
    for epoch in range(initial_epoch,num_epochs):
        model.config.steps = epoch

        train_loss, _ = model.train(train_loader)
        val_loss, _ = model.test(val_loader)

        losses['train_loss'].append(train_loss)
        losses['val_loss'].append(val_loss)
        wandb.log({ 
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(output_path + '/vqvae_UKBB_best.pt', epoch, best_val_loss, losses)
            print(f"Epoch {epoch} is current the best")

        with open(output_path + '/losses.pkl', 'wb') as file:
            pickle.dump(losses, file)
        
        model.save_checkpoint(output_path + '/checkpoint.pt',epoch,best_val_loss,losses)
        
    model.save_checkpoint(output_path + '/vqvae_UKBB_final.pt', epoch, best_val_loss, losses)