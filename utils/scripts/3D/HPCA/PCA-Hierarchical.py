import pickle
import os
import numpy as np
from scipy.linalg import eigh 
import sys
from sklearn.decomposition import PCA


#pca_path = '/work/forkert_lab/erik/PCA/slices-z'
#reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'
output_path = '/work/forkert_lab/erik/PCA3D/HPCA_10'
number_eigh = 1500
new_input_data_path = '/work/forkert_lab/erik/MACAW/encoded/slices_10'

max_slices = 182
slices = 100
# Get the center slices
z_init = round((max_slices-slices)/2)
z_end = max_slices - z_init
slices_per_model = 10
slice_overlap = 5
'''
max_slices = 182
slices = int(sys.argv[1])
# Get the center slices
z_init = round((max_slices-slices)/2)
z_end = max_slices - z_init


all_evecs = []
first = True

for slice_number in range(z_init,z_end):
    data = np.load(reshaped_path + '/reshaped_slice_{}.npy'.format(slice_number))
    with open(pca_path + '/evecs_test_slice_{}.pkl'.format(slice_number),'rb') as f:  
        evec = pickle.load(f)
        evec = evec[:,:number_eigh]
        encoded = np.matmul(data,evec)
        if first == True:
            all_evecs = encoded
            first = False
        else:
            all_evecs = np.concatenate((all_evecs,encoded), axis=1)

print("It has to be false to be sure that was concatenated")
print(first)
print("PCAs were loaded and concatenated")

with open(new_input_data_path + f"/encoded_{number_eigh}_perslice.npy", 'wb') as f2:
    np.save(f2, all_evecs)
print("Saved the encoded matrix")
'''

for slice_number in range(z_init,z_end-slice_overlap,slice_overlap):
    all_evecs = np.load(new_input_data_path + f"/encoded_1500_perslice_{slice_number}.npy")
    #print(len(all_evecs))
    #all_evecs = np.stack([d for d in all_evecs], axis=0)
    #print("PCAs were stacked")
    print(all_evecs.shape)
    #all_evecs = all_evecs.reshape(all_evecs.shape[0],-1)
    #print("PCAs were reshaped")
    #print(all_evecs.shape)

    pca = PCA(svd_solver='randomized')
    pca.fit(all_evecs)

    #evecs = np.matmul(all_evecs.T , all_evecs)
    #print("cov_matrix calcuated")
    #_, evecs = eigh(evecs, eigvals=(0,all_evecs.shape[1]-1))
    #print("evec calculated")
    #evecs = evecs[::,::-1]

    with open(output_path + f"/variance_evecs_HPCA_{number_eigh}_{slice_number}.npy", 'wb') as f_variance:
        np.save(f_variance, np.array(pca.explained_variance_ratio_))

    with open(output_path + f"/evecs_HPCA_{number_eigh}_{slice_number}.pkl", 'wb') as filefinal:
        pickle.dump(pca.components_, filefinal)


