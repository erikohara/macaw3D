import pickle
import numpy as np
import sys


pca_path = '/work/forkert_lab/erik/PCA/slices-z-sklearn'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'
number_eigh = 1000
new_input_data_path = '/work/forkert_lab/erik/MACAW/encoded/sklearn'
residuals_path = '/work/forkert_lab/erik/MACAW/encoded/residuals'



max_slices = 182
slices = 100
# Get the center slices
z_init = round((max_slices-slices)/2)
z_end = max_slices - z_init


all_evecs = []
first = True


for slice_number in range(z_init,z_end):
    data = np.load(reshaped_path + '/reshaped_test_slice_{}.npy'.format(slice_number))
    with open(pca_path + '/evecs_slice_{}.pkl'.format(slice_number),'rb') as f:  
        evec = pickle.load(f)
        evec = evec[:number_eigh]
        encoded = np.matmul(data,evec.T)
        residuals = data - np.matmul(encoded,evec)
        with open(residuals_path + f"/residuals_slice_{slice_number}.npy", 'wb') as f2:
            np.save(f2, residuals)
        if first == True:
            all_evecs = encoded
            first = False
        else:
            all_evecs = np.concatenate((all_evecs,encoded), axis=1)

print("It has to be false to be sure that was concatenated")
print(first)
print("PCAs were loaded and concatenated")

'''
with open(new_input_data_path + f"/encoded_{number_eigh}_perslice-test.npy", 'wb') as f2:
    np.save(f2, all_evecs)
print("Saved the encoded matrix")
'''