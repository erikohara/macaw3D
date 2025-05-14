import torch
import sys
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
crop_size = (102,150,150)
n_sample = 8

ukbb_path_T1_test = '/work/forkert_lab/erik/T1_warped/test'
df_ukbb_test = '/home/erik.ohara/UKBB/test.csv'
model_path = '/work/forkert_lab/erik/MACAW/models/AE3D_10125_fast'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/3D'
output_path = '/work/forkert_lab/erik/MACAW/decoded_images/AE3D_10125_fast'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# random sample to test
df_test = pd.read_csv(df_ukbb_test,low_memory=False)
indexes = df_test.index[df_test['eid'].isin([1826719,
                                   2246047,
                                   2444112,
                                   2590999,
                                   2658672,
                                   3313466,
                                   3542536,
                                   5485166])].tolist()
#df_test = df_test.sample(n=n_sample, random_state=1)

#dataset_test = UKBBT13DDataset(df_test,ukbb_path_T1_test, transforms.Compose([ToFloatUKBB(),ToTensor(), Crop3D(crop_size)]))
data_test = np.load(reshaped_path + '/reshaped_3D_102_150_150_train.npy')
print(f"Data test loaded: {data_test.shape}")

d_obs = data_test[indexes,:,:,:,:]  
df_test = df_test.iloc[indexes] 
data_test = data_test[indexes,:]

print(f"sample data test: {data_test.shape}")


dataset_test = CustomDataset(data_test.astype(np.float32), device)
test_loader = DataLoader(dataset_test, batch_size=batch_size)

ae = torch.load(model_path + '/ae_UKBB_best.pt',map_location=torch.device('cpu'))
ae.device = device

encoded_data = ae.encode(test_loader)

imgs = DataLoader(CustomDataset(encoded_data), batch_size=batch_size)

decoded_data = ae.decode(imgs)
print(f"decoded_data.shape: {decoded_data.shape}")


'''
loss_func = torch.nn.MSELoss()
data_test = torch.from_numpy(data_test)
decoded_data = torch.from_numpy(decoded_data)
loss = loss_func(data_test,decoded_data)
print(f"The loss is {loss}")
'''
for idx,each_image in enumerate(decoded_data):
     # Getting the orifinal affine and header
    image_path = ukbb_path_T1_test + '/' + str(int(df_test.iloc[idx]["eid"])) + '.nii.gz'
    original_image = nib.load(image_path)
    # Saving new generated images
    each_image = each_image[0,:,:,:]
    each_image = np.swapaxes(each_image,0,2)
    each_image = np.swapaxes(each_image,0,1)
    ni_img = nib.Nifti1Image(each_image, original_image.affine, original_image.header)
    nib.save(ni_img, output_path + "/" + str(int(df_test.iloc[idx]["eid"])) + '.nii.gz')