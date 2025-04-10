#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import sys
import torch
from  torch.utils.data import  DataLoader
import torch.distributions as td
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
macaw_path = '/home/erik.ohara/macaw'
sys.path.append(macaw_path +'/')
from utils.helpers import dict2namespace
from utils.datasets import CustomDataset
from macaw import MACAW

slice_initial = 41
slice_final = 141
n_slices = slice_final - slice_initial
nevecs = 50
ncauses = 2
ncomps = 200
nbasecomps = 25
ae_batch_size = 2
ukbb_path = '/home/erik.ohara/UKBB'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'
model_path = f"/work/forkert_lab/erik/MACAW/models/macaw_AE2D_200"
ae_path = f'/work/forkert_lab/erik/MACAW/models/AE2D_200'
original_images = "/work/forkert_lab/erik/T1_warped"
output_image = f'/work/forkert_lab/erik/MACAW/cf_images/macaw_AE2D_200_zero_diff'

print("Starting code")
if not os.path.exists(output_image):
    os.makedirs(output_image)

all_data = []
all_data_dim = []
for z_slice in tqdm(range(slice_initial,slice_final)):
    data = np.load(reshaped_path + '/reshaped_test_slice_{}.npy'.format(z_slice))
    data = data.reshape(data.shape[0],1,150,150)
    all_data.append(data)
    all_data_dim.append(data.shape[1])


data_path = ukbb_path + '/ukbb_img.csv'
df = pd.read_csv(data_path,low_memory=False)
print(f"The original size of the dataframe is {df.shape}")

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

#test set
data_path = ukbb_path + '/test.csv'

df = pd.read_csv(data_path,low_memory=False)
df.sort_values(by=['eid'], inplace=True)
print(f"The original size of the dataframe is {df.shape}")

all_eid = df[['eid']].to_numpy()
min_age = df['Age'].min()
print(f"Age min: {min_age}")
sex = df['Sex'] 
age = df['Age'] - min_age

nsamples = len(df)
sex_obs = sex
age_obs = age
random_eids = all_eid[:,0]

# counterfactual defition
#cf_vals = {1:cf_age_definition - min_age}

ae_models = []
all_c_cf = [np.zeros((nsamples,ncomps)) for _ in range(slice_initial,slice_final)]
for z_slice in range(slice_initial,slice_final):
    order_slice = z_slice - slice_initial
    dataset_test = CustomDataset(all_data[order_slice].astype(np.float32), config.device)
    test_loader = DataLoader(dataset_test, batch_size=ae_batch_size)
    ae = torch.load(ae_path + f'/slice_{z_slice}/ae_UKBB_best.pt',map_location=torch.device(config.device))
    ae.device = config.device
    ae_models.append(ae)
    encoded_data_all = ae.encode(test_loader)
    datashape1 = ncauses + nevecs
    for ev in range(0,ncomps-nbasecomps,nevecs-nbasecomps):
        encoded_obs =  encoded_data_all[:,ev:ev+nevecs] 
        try:
            macaw = MACAW.MACAW(config)
            macaw.load_model(model_path + f'/slice_{z_slice}/macaw_ukbb_PCA3D_{ev}.pt', edges, priors, datashape1)
            macaw.model.eval()
            with torch.no_grad():
                X_obs = np.hstack([np.array(sex_obs)[:,np.newaxis], np.array(age_obs)[:,np.newaxis], encoded_obs])
                z_obs = macaw._forward_flow(X_obs) 
                cc = macaw._backward_flow(z_obs)
        except FileNotFoundError as e:
            print(e)
            print(f"File not finded: /slice_{z_slice}/macaw_ukbb_PCA3D_{ev}.pt")
            cc[:,ncauses:] = encoded_obs.copy()
            #d_encoded_cf = encoded_obs
        except ValueError as value_error:
            print(f"IDs of subjects: {random_eids}")
            print(cc.shape)
            print(np.isinf(cc).nonzero())
            print(f"Error in model /slice_{z_slice}/macaw_ukbb_PCA3D_{ev}.pt")
            raise Exception('Just to stop and see', value_error)
        all_c_cf[order_slice][:,ev:ev+nevecs] = cc[:,ncauses:]

all_c_cf_2 = np.array(all_c_cf)
print(f"Shape do all_c_cf: {all_c_cf_2.shape}")

nan_places = np.argwhere(np.isnan(all_c_cf_2))
print("Positions of nan values")
print(nan_places)
print("")

sex_cf = ['Male' if round(s) else 'Female' for s in cc[:,0]]
age_cf = age_obs + min_age

all_decoded_cf = []
for z_slice in tqdm(range(slice_initial,slice_final)):
    order_slice = z_slice - slice_initial
    imgs = DataLoader(CustomDataset(all_c_cf_2[order_slice].astype(np.float32)), batch_size=ae_batch_size)
    decoded_cf = ae_models[order_slice].decode(imgs)
    all_decoded_cf.append(decoded_cf)

import nibabel as nib

all_decoded_cf_2 = np.array(all_decoded_cf)
print(f"Shape do all_decoded_cf: {all_decoded_cf_2.shape}")

def count_files_folder(path_file_to_count):
    return len([name for name in os.listdir(path_file_to_count) if os.path.isfile(os.path.join(path_file_to_count, name))])

number_files_output = count_files_folder(output_image)

for individual in range(nsamples):
    numpy_image = np.array([])
    for z_slice in range(slice_initial,slice_final):
        order_slice = z_slice - slice_initial
        if order_slice == 0:
            numpy_image = all_decoded_cf[order_slice][individual].reshape(150,150)
        else:
            numpy_image = np.dstack((numpy_image,all_decoded_cf[order_slice][individual].reshape(150,150)))
    
    image_path = original_images + '/test/' + str(random_eids[individual]) + '.nii.gz'
    original_image = nib.load(image_path)
    ni_img = nib.Nifti1Image(numpy_image, original_image.affine, original_image.header)
    nib.save(ni_img, output_image + "/" + str(random_eids[individual])  + f"_{int(age_cf[individual])}_{sex_cf[individual]}_{slice_initial}-{slice_final-1}.nii.gz")
    if count_files_folder(output_image) > number_files_output:
        number_files_output = count_files_folder(output_image)
    else:
        print(f"File {random_eids[individual]} not saved for some reason")
        print(numpy_image.shape)
        print(np.argwhere(np.isnan(numpy_image)))
        break

print(f"Number of unique ID files is: {count_files_folder(output_image)}")