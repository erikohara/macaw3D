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
import argparse
os.environ['TF_DETERMINISTIC_OPS'] = '1'

macaw_path = '/home/erik.ohara/macaw/'
sys.path.append(macaw_path)
from utils.helpers import dict2namespace 
from compression.vqvae import vqvae
from utils.datasets import UKBBT13DDataset, CustomDataset
from utils.customTransforms import ToFloatUKBB, Crop3D

parser = argparse.ArgumentParser()
#parser.add_argument('encoded_dim')
parser.add_argument('z_slice')
args = parser.parse_args()
#encoded_dim = int(args.encoded_dim)
z_slice = args.z_slice

with open(macaw_path +'/compression/vqvae/vqvae2D.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
    
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_epochs = config.training.epochs

df_ukbb_train = '/home/erik.ohara/UKBB/train.csv'
df_ukbb_val = '/home/erik.ohara/UKBB/val.csv'
output_path = f'/work/forkert_lab/erik/MACAW/models/vqvae2D_{config.vq.hidden_size}/slice_{z_slice}'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'
load_checkpoint = False  

if not os.path.exists(output_path):
    os.makedirs(output_path)


# In[12]:
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_train = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(z_slice)) 
data_train = data_train.reshape(data_train.shape[0],1,150,150)
print(f"Data train loaded: {data_train.shape}")

data_val = np.load(reshaped_path + '/reshaped_val_slice_{}.npy'.format(z_slice)) 
data_val = data_val.reshape(data_val.shape[0],1,150,150)
print(f"Data val loaded: {data_val.shape}")

dataset_train = CustomDataset(data_train.astype(np.float32), device)
dataset_val = CustomDataset(data_val.astype(np.float32), device)

train_loader = DataLoader(dataset_train, shuffle=True, batch_size=config.training.batch_size)
val_loader = DataLoader(dataset_val, shuffle=True, batch_size=config.training.batch_size)



writer = SummaryWriter(macaw_path + 'logs/vqvae_MM')

wandb_config = {
        "learning_rate": config.optim.lr,
        "batch_size": config.training.batch_size,
        'hidden_size': config.vq.hidden_size,
        "epochs": num_epochs,
        'output_path': output_path,
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'patience': config.training.patience,
        'z_slice': z_slice
    }

with wandb.init(project="VQ-VAE_2D_training", config=wandb_config):

    model = vqvae.VQVAE(config,writer)
    losses = {'train_loss':[],'val_loss':[]}
    best_val_loss = 99999999

    initial_epoch = 0

    if load_checkpoint:
        initial_epoch, losses, best_val_loss = model.load_checkpoint(output_path + '/checkpoint.pt')


    print(f"Start training from epoch {initial_epoch} with best loss at {best_val_loss}")
    epochs_no_improving = 0
    best_epoch = 0
    patience = config.training.patience
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
        '''
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        '''

        with open(output_path + '/losses.pkl', 'wb') as file:
            pickle.dump(losses, file)
        
        model.save_checkpoint(output_path + '/checkpoint.pt',epoch,best_val_loss,losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improving = 0
            model.save_checkpoint(output_path + '/vqvae_UKBB_best.pt', epoch, best_val_loss, losses)
            print(f"Epoch {epoch} is current the best")
        elif patience!=0:
            if epochs_no_improving == patience:
                print(f"Best epoch was epoch {best_epoch} +1 with a Val losss of {best_val_loss:.3f}")
                break
            else:
                epochs_no_improving += 1
        
    model.save_checkpoint(output_path + '/vqvae_UKBB_final.pt', epoch, best_val_loss, losses)