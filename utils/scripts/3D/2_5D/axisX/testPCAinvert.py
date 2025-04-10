import torch
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

sys.path.append('/home/erik.ohara/macaw/')
from utils.customTransforms import ToFloatUKBB, Crop3D
from utils.datasets import UKBBT13DDataset, CustomDataset

slice_initial = 16
slice_final = 166
batch_size = 4
#nevecs = 10625

reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-x'
evec_path = '/work/forkert_lab/erik/PCA/slices-x'
output_path = '/work/forkert_lab/erik/JMI_paper_2/revisions'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


all_data = []
all_data_dim = []
for z_slice in range(slice_initial,slice_final):
    data = np.load(reshaped_path + '/reshaped_val_slice_{}.npy'.format(z_slice))
    all_data.append(data)
    all_data_dim.append(data.shape[1])
all_data = np.array(all_data)
all_data_torch = torch.from_numpy(all_data)

all_evecs = []
for z_slice in range(slice_initial,slice_final):
    with open(evec_path + '/evecs_slice_{}.pkl'.format(z_slice),'rb') as f:  
        evecs = pickle.load(f)
        all_evecs.append(evecs)



def encode(data, evecs):
    return np.matmul(data,evecs.T)

def decode(data,evecs):
    return np.matmul(data,evecs)

df = pd.DataFrame(columns=["nevecs", "MSEloss"])

for nevecs in range(100,3000,100):
    all_decoded = []
    for z_slice in range(slice_initial,slice_final):
        order_slice = z_slice - slice_initial
        encoded_obs = encode(all_data[order_slice],all_evecs[order_slice][:nevecs])  
        all_decoded.append(decode(encoded_obs,all_evecs[order_slice][:nevecs])  )
    #print(eids)
    all_decoded = np.array(all_decoded)

    loss_func = torch.nn.MSELoss()

    all_decoded = torch.from_numpy(all_decoded)
    #print(f"image_data.shape: {image_data.shape}")
    #print(f"decoded_data.shape: {decoded_data_torch.shape}")
    loss = loss_func(all_data_torch,all_decoded)
    print(f"Nevecs {nevecs}: The MSE loss is {loss}")
    df.loc[len(df)] = {"nevecs": nevecs, "MSEloss": loss.item()}

df.to_csv(f"{output_path}/PCA_2Dsagittal_MSE_per_nevecs.csv")
