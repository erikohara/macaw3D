#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import sys
import yaml
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('/home/erik.ohara/macaw/')
from utils.helpers import dict2namespace 
from compression.vqvae import vqvae
from utils.datasets import CustomDataset

reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
df_ukbb_test = '/home/erik.ohara/UKBB/test.csv'
macaw_path = '/home/erik.ohara/macaw/'
output_path = '/work/forkert_lab/erik/MACAW/decoded_images/vqvae3D_8'
model_path = '/work/forkert_lab/erik/MACAW/models/vqvae3D_8'
ukbb_path_T1_test = '/work/forkert_lab/erik/T1_warped/test'
n_samples = 8
batch_size = 4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_test = np.load(reshaped_path + '/reshaped_3D_102_150_150_test.npy')
print("Data test loaded")
print(f"data_test.shape: {data_test.shape}")

df_test = pd.read_csv(df_ukbb_test,low_memory=False)
'''
indexes = df_test.index[df_test['eid'].isin([ 1052595,
                                    1826719,
                                   2246047,
                                   2444112,
                                   2590999,
                                   2658672,
                                   3313466,
                                   3542536,
                                   5485166])].tolist()
'''

#rands = np.random.randint(0,data_test.shape[0],n_samples)
#print(rands)
d_obs = data_test[:,:,:,:,:]  
#df_test = df_test.iloc[indexes] 
#print(df_test_rand)
#c_obs = causes[rands,:]
_,_,z_axis, y_axis, x_axis = d_obs.shape

dataset_test = CustomDataset(d_obs.astype(np.float32), device)

test_loader = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

with open(macaw_path +'/compression/vqvae/vqvae.yaml', 'r') as f:
    config_raw = yaml.load(f, Loader=yaml.FullLoader)
    
config = dict2namespace(config_raw)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_epochs = config.training.epochs

writer = SummaryWriter(macaw_path + 'logs/vqvae_MM')

model = vqvae.VQVAE(config,writer)
model.load_checkpoint(model_path + '/vqvae_UKBB_best.pt')

encoded_data = model.encode(test_loader)
print(f"encoded_data.shape: {encoded_data.shape}")
z_axis = int(z_axis/6)
y_axis = int(y_axis/6)
x_axis = int(x_axis/6)
print(z_axis, y_axis, x_axis)
encoded_data = encoded_data.reshape(encoded_data.shape[0], z_axis, y_axis, x_axis)
decoded_data = model.decode(encoded_data)

print(f"decoded_data.shape: {decoded_data.shape}")


loss_func = torch.nn.MSELoss()
data_test = torch.from_numpy(data_test)
decoded_data = torch.from_numpy(decoded_data)
loss = loss_func(data_test[:,0,:,:,:],decoded_data)
print(f"The loss is {loss}")
'''
for idx,each_image in enumerate(decoded_data):
     # Getting the orifinal affine and header
    image_path = ukbb_path_T1_test + '/' + str(int(df_test.iloc[idx]["eid"])) + '.nii.gz'
    original_image = nib.load(image_path)
    # Saving new generated images
    each_image = np.swapaxes(each_image,0,2)
    each_image = np.swapaxes(each_image,0,1)
    ni_img = nib.Nifti1Image(each_image, original_image.affine, original_image.header)
    nib.save(ni_img, output_path + "/" + str(int(df_test.iloc[idx]["eid"])) + '.nii.gz')
'''