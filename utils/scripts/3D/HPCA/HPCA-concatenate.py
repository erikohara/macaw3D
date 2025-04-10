import pickle
import numpy as np
import sys


pca_path = '/work/forkert_lab/erik/PCA/slices-z-sklearn'
reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'
number_eigh = 1500
new_input_data_path = '/work/forkert_lab/erik/MACAW/encoded/slices_10'



max_slices = 182
slices = 100
# Get the center slices
z_init = round((max_slices-slices)/2)
z_end = max_slices - z_init
slices_per_model = 10
slice_overlap = 5





for slice_number in range(z_init,z_end-slice_overlap,slice_overlap):
    print(slice_number)
    all_evecs = []
    first = True
    for slice_part in range(slices_per_model):
        data = np.load(reshaped_path + f"/reshaped_slice_{slice_number+slice_part}.npy")
        with open(pca_path + '/evecs_slice_{}.pkl'.format(slice_number+slice_part),'rb') as f:  
            evec = pickle.load(f)
            evec = evec[:number_eigh]
            encoded = np.matmul(data,evec.T)
            if first == True:
                all_evecs = encoded
                first = False
            else:
                all_evecs = np.concatenate((all_evecs,encoded), axis=1)
    with open(new_input_data_path + f"/encoded_{number_eigh}_perslice_{slice_number}.npy", 'wb') as f2:
        np.save(f2, all_evecs)
