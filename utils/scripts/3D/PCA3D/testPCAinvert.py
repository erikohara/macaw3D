import torch
import sys
import pickle
import nibabel as nib
import pandas as pd
import numpy as np

from  torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

sys.path.append('/home/erik.ohara/macaw/')
from utils.customTransforms import ToFloatUKBB, Crop3D
from utils.datasets import UKBBT13DDataset, CustomDataset

batch_size = 4
crop_size = (100,150,150)
n_sample = 8
#nevecs = 10625

ukbb_path_T1_test = '/work/forkert_lab/erik/T1_warped/val'
df_ukbb_test = '/home/erik.ohara/UKBB/val.csv'
evec_path = '/work/forkert_lab/erik/PCA3D'
output_path = '/work/forkert_lab/erik/JMI_paper_2/revisions'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# random sample to test
df_test = pd.read_csv(df_ukbb_test,low_memory=False)
#df_test = df_test.sample(n=n_sample, random_state=1)
'''
indexes = df_test.index[df_test['eid'].isin([1826719,
                                   2246047,
                                   2444112,
                                   2590999,
                                   2658672,
                                   3313466,
                                   3542536,
                                   5485166])].tolist()
df_test = df_test.iloc[indexes, :]
'''

dataset_test = UKBBT13DDataset(df_test,ukbb_path_T1_test, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
test_loader = DataLoader(dataset_test, batch_size=batch_size)

with open(evec_path + "/evecs.pkl",'rb') as f:  
    evecs3D = pickle.load(f)
print("PCA loaded")

image_data = []
eids = []
data_all = [(d.numpy(),eid) for d,_,_,_,eid in test_loader]
for each_data in data_all:
    image_data.append(each_data[0])
    eids.append(each_data[1])
image_data = np.concatenate(image_data,axis=0)
eids = np.concatenate(eids,axis=0)
print(f"image_data.shape: {image_data.shape}")

image_data = image_data.reshape(image_data.shape[0],-1)
image_data = torch.from_numpy(image_data)
print(image_data.shape)

def encode(data, evecs):
    return np.matmul(data,evecs.T)

def decode(data,evecs):
    return np.matmul(data,evecs)

df = pd.DataFrame(columns=["nevecs", "MSEloss"])

for nevecs in range(1000,18000,1000):
    encoded_data = encode(image_data,evecs3D[:nevecs])  

    #print(eids)

    decoded_data = decode(encoded_data,evecs3D[:nevecs])  

    loss_func = torch.nn.MSELoss()

    #decoded_data_torch = torch.from_numpy(decoded_data)
    #print(f"image_data.shape: {image_data.shape}")
    #print(f"decoded_data.shape: {decoded_data_torch.shape}")
    loss = loss_func(image_data,decoded_data)
    print(f"Nevecs {nevecs}: The MSE loss is {loss}")
    df.loc[len(df)] = {"nevecs": nevecs, "MSEloss": loss.item()}

df.to_csv(f"{output_path}/PCA3D_MSE_per_nevecs.csv")
'''
for idx,each_image in enumerate(decoded_data):
     # Getting the orifinal affine and header
    image_path = ukbb_path_T1_test + '/' + eids[idx]
    original_image = nib.load(image_path)
    # Saving new generated images
    each_image = each_image.reshape(100,150,150)
    each_image = np.swapaxes(each_image,0,1)
    each_image = np.swapaxes(each_image,1,2)
    ni_img = nib.Nifti1Image(each_image, original_image.affine, original_image.header)
    nib.save(ni_img, output_path + "/" + eids[idx])
'''