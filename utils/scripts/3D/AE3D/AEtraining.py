#!/usr/bin/env python
# coding: utf-8

import sys
import torch
import pickle
import wandb
import argparse
import pandas as pd
import numpy as np

from  torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

sys.path.append('/home/erik.ohara/macaw/')
from compression.autoencoder.AE3D import AE3D
from utils.datasets import UKBBT13DDataset, CustomDataset
from utils.customTransforms import ToFloatUKBB, Crop3D

parser = argparse.ArgumentParser()
parser.add_argument('encoded_dim')
parser.add_argument('wandb_run_id')
args = parser.parse_args()
encoded_dim = int(args.encoded_dim)
wandb_run_id = args.wandb_run_id


batch_size = 8
num_epochs = 300
#crop_size = (100,148,148)
#encoded_dim = 10125
#ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/train'
#ukbb_path_T1_val = '/work/forkert_lab/erik/T1_warped/val'
#df_ukbb_train = '/home/erik.ohara/UKBB/train.csv'
#df_ukbb_val = '/home/erik.ohara/UKBB/val.csv'
output_path = f'/work/forkert_lab/erik/MACAW/models/AE3D_{encoded_dim}_previous'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
load_checkpoint = False  
learning_rate = 1e-6
patience = 15

if not os.path.exists(output_path):
    os.makedirs(output_path)

wandb_config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        'encoded_dim': encoded_dim,
        "epochs": num_epochs,
        'output_path': output_path,
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'patience': patience
    }
with wandb.init(project="AEtraining_previous", config=wandb_config): #, id=wandb_run_id, resume="must"):

    # In[12]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_train = np.load(reshaped_path + '/reshaped_3D_102_150_150_train.npy')
    #data_train = data_train[:,:,:,2:-2,2:-2]
    print(f"Data train loaded: {data_train.shape}")

    data_val = np.load(reshaped_path + '/reshaped_3D_102_150_150_val.npy')
    #data_val = data_val[:,:,:,2:-2,2:-2]
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
    val_loader = DataLoader(dataset_val, batch_size=batch_size)
    '''

    losses = {'train_loss':[],'val_loss':[]}
    ae = AE3D(encoded_dim, val_loader,lr=learning_rate)
    best_val_loss = 99999999

    initial_epoch = 0
    if load_checkpoint:
        initial_epoch, losses, best_val_loss = ae.load_checkpoint(output_path + '/checkpoint.pt')

    print(f"Start training from epoch {initial_epoch} with best loss at {best_val_loss}")
    epochs_no_improving = 0
    best_epoch = 0
    for epoch in range(initial_epoch,num_epochs):
        print('\n Starting EPOCH {}/{}'.format(epoch + 1, num_epochs))
        train_loss = ae.train(train_loader)
        val_loss = ae.test(val_loader)

        losses['train_loss'].append(train_loss)
        losses['val_loss'].append(val_loss)
        wandb.log({ 
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        print('EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improving = 0
            torch.save(ae, output_path + '/ae_UKBB_best.pt')
            print(f"Epoch {epoch+1} is current the best")
        elif patience!=0:
            if epochs_no_improving == patience:
                print(f"Best epoch was epoch {best_epoch} +1 with a Val losss of {best_val_loss:.3f}")
                break
            else:
                epochs_no_improving += 1

        with open(output_path + '/losses.pkl', 'wb') as file:
            pickle.dump(losses, file)
        
        ae.save_checkpoint(output_path + '/checkpoint.pt',epoch+1,best_val_loss,losses)
    torch.save(ae, output_path + '/ae_UKBB_final.pt')