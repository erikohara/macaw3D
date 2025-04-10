#!/usr/bin/env python
# coding: utf-8

# In[10]:

import sys
import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import yaml

macaw_path = '/home/erik.ohara/macaw'
sys.path.append(macaw_path +'/')

from utils.helpers import dict2namespace

import pickle
from tqdm.notebook import tqdm
import pandas as pd
import nibabel as nib


#nevecs = 500
#ncauses = 2
#ncomps = 25000
nevecs = 50
ncauses = 2
ncomps = 18000
nbasecomps = 25
ukbb_path = '/home/erik.ohara/UKBB'
evec_path_3D = '/work/forkert_lab/erik/PCA3D/HPCA_sklearn' 
evec_path = '/work/forkert_lab/erik/PCA/slices-z-sklearn'
original_images = "/work/forkert_lab/erik/T1_warped"
ukbb_path_T1_slices = '/work/forkert_lab/erik/T1_warped/test'
model_path = '/work/forkert_lab/erik/MACAW/models/HPCA_18000'
input_path = '/work/forkert_lab/erik/MACAW/encoded/sklearn'
output_image = f"/work/forkert_lab/erik/MACAW/cf_images/HPCA_18000_zero"
number_images = 2370
cf_age_definition = 55
number_slices = 100
slice_initial = 41
slice_final = 141

data_test = np.load(input_path + '/encoded_1000_perslice-test.npy')
data_dim = data_test.shape[1]
print("Data loaded")

data_train = np.load(input_path + '/encoded_1000_perslice.npy')
print("Data loaded")

# Loading evecs for each slice
all_evecs = []
for z_slice in range(slice_initial,slice_final):
    with open(evec_path + '/evecs_slice_{}.pkl'.format(z_slice),'rb') as f:
        evecs = pickle.load(f)
        all_evecs.append(evecs)

with open(evec_path_3D + "/evecs_HPCA_1000_3D_41_141.pkl",'rb') as f:  
    evecs = pickle.load(f)
print("PCA loaded")
evecs = evecs[:]

# Create output folder
if not os.path.exists(output_image):
    os.makedirs(output_image)

def encode(data, evecs):
    return np.matmul(data,evecs.T)

def decode(data,evecs):
    return np.matmul(data,evecs)

# Loading subjects data
data_path = ukbb_path + '/ukbb_img.csv'
df = pd.read_csv(data_path,low_memory=False)
#df['Sex'] = df['Sex'] + np.random.rand(df.shape[0]) - 0.5
#scaler_a = StandardScaler()
#df['Age'] = scaler_a.fit_transform(df[['Age']])

print(f"The original size of the dataframe is {df.shape}")

# getting only the test set
df_test = pd.DataFrame(columns=df.columns)
for each_file in os.listdir(ukbb_path_T1_slices):
    if '.nii' in each_file:
        file_id = each_file.split('.nii')[0]
        df_test = pd.concat([df[df['eid'] == int(file_id)],df_test.loc[:]]).reset_index(drop=True)
df_test.sort_values(by=['eid'], inplace=True)

df = df_test
print(f"The size of the dataframe (just test set) is {df.shape}")

min_age = df['Age'].min()
all_eid = df[['eid']].to_numpy()
sex = df['Sex'] 
age = df['Age'] - min_age


# causal Graph
sex_to_latents = [(0,i) for i in range(ncauses,nevecs+ncauses)]
age_to_latents = [(1,i) for i in range(ncauses,nevecs+ncauses)]
autoregressive_latents = [(i,j) for i in range(ncauses,nevecs+ncauses-1) for j in range(i+1,nevecs+ncauses)]
edges = sex_to_latents + age_to_latents + autoregressive_latents
#causes = df[['Age','Sex']].to_numpy()
#edges = [(0,i)for i in range(ncauses,nevecs)]+[(1,i)for i in range(ncauses,nevecs)] + [(i,j) for i in range(ncauses,nevecs+ncauses-1) for j in range(i+1,nevecs+ncauses)]

# Loading Config
with open(macaw_path + '/config/ukbb.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


'''
# Enconding data
scalers = {}
for e in range(0,ncomps,nevecs):
    encoded_data =  encode3D(data_train,evecs[e:e+nevecs,:])
    scaler = StandardScaler()
    encoded_data = scaler.fit_transform(encoded_data)
    scalers[f"{e}"] = scaler
'''


# Random subjects Counterfactuals
#cf_vals = {0:cf_age_definition}
#cf_vals = {1:cf_age_definition - min_age}
nsamples = number_images
rands = np.random.randint(0,data_test.shape[0],nsamples)    
#c_obs = causes[rands,:]
sex_obs = sex[rands]
age_obs = age[rands]
d_obs = data_test[rands,:]

# Generating the CF
#d_encodes = []
s=0
first_ev = True
c_cf = np.zeros((nsamples,ncomps))
for ev in range(0,ncomps-nbasecomps,nevecs-nbasecomps):
#for ev in range(0,ncomps,nevecs):
    encoded_obs =  encode(d_obs,evecs[ev:ev+nevecs,:])
    try:
        macaw = torch.load(model_path + f'/{nevecs}/macaw_ukbb_HPCA_{ev}.pt')
        #scaler = scalers[f"{ev}"]

        #encoded_obs = scaler.transform(encoded_obs)
        #X_obs = np.hstack([c_obs,encoded_obs])   

        #cc = macaw.counterfactual(X_obs,cf_vals)
        macaw.model.eval()
        with torch.no_grad():
            X_obs = np.hstack([sex_obs[:,np.newaxis], age_obs[:,np.newaxis], encoded_obs]) 
            z_obs = macaw._forward_flow(X_obs)
            cc = macaw._backward_flow(z_obs)
        #d_encoded_cf = cf[:,ncauses:]
        #c_cf = cf[:,:ncauses]
        '''
        if first_ev:
            c_cf = cf[:,:ncauses]
            first_ev = False
        elif not np.isnan(cf[:,:ncauses]).any():
            c_cf = cf[:,:ncauses]
        '''
        #d_encoded_cf = scaler.inverse_transform(d_encoded_cf)
    except FileNotFoundError as e:
        print(e)
        cc[:,ncauses:] = X_obs[:,ncauses:].copy()
        #d_encoded_cf = encoded_obs
    #d_encodes.append(d_encoded_cf)
    c_cf[:,ev:ev+nevecs] = cc[:,ncauses:]
    s+=1

#age = scaler_a.inverse_transform(c_obs[:,0:1])
#sex = ['Male' if round(s) else 'Female' for s in c_obs[:,1]]
#titles = [f'Age:{a}, Sex:{s}' for a,s in zip(age,sex)]
age_obs = age_obs + min_age
sex_obs = ['Male' if round(s) else 'Female' for s in sex_obs]
titles = [f'Age:{a}, Sex:{s}' for a,s in zip(age_obs,sex)]

sex_cf = ['Male' if round(s) else 'Female' for s in cc[:,0]]
age_cf = cc[:,1] + min_age
titles_cf = [f'Age:{np.round(a)}, Sex:{s}' for a,s in zip(age_cf,sex_cf)]

#print(f"Shape of c_cf: {c_cf.shape}")
#encoded_cf = np.hstack(c_cf)
#print(f"Shape of encoded_cf: {encoded_cf.shape}")
decoded_cf = decode(c_cf,evecs[:ncomps,:])

all_decoded_cf = []
slice_order = 0
for ev in range(0,data_dim,1000):
    slice_decoded_cf = decode(decoded_cf[:,ev:ev+1000],all_evecs[slice_order][0:1000])
    all_decoded_cf.append(slice_decoded_cf)
    slice_order += 1
print(f"Decoded for {slice_order} slices")


'''
import nibabel as nib

random_eids = all_eid[rands,0]
for individual in range(nsamples):
    numpy_image = decoded_cf[individual].reshape(150,150,100)
    ni_img = nib.Nifti1Image(numpy_image, affine=np.eye(4))
    nib.save(ni_img, output_image + "/" + str(random_eids[individual])  + f"_{int(age_cf[individual][0])}_{sex_cf[individual][0]}_HPCA.nii.gz")
'''

random_eids = all_eid[rands,0]
for individual in range(nsamples):
    numpy_image = np.array([])
    for z_slice in range(slice_initial,slice_final):
        order_slice = z_slice - slice_initial
        if order_slice == 0:
            numpy_image = all_decoded_cf[order_slice][individual].reshape(150,150)
        else:
            numpy_image = np.dstack((numpy_image,all_decoded_cf[order_slice][individual].reshape(150,150)))
    #numpy_image = np.swapaxes(numpy_image,0,1)
    # Getting the orifinal affine and header
    image_path = original_images + '/test/' + str(random_eids[individual]) + '.nii.gz'
    original_image = nib.load(image_path)
    # Saving new generated images
    ni_img = nib.Nifti1Image(numpy_image, original_image.affine, original_image.header)
    nib.save(ni_img, output_image + "/" + str(random_eids[individual])  + ".nii.gz")
