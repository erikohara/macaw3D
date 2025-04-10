import os

path = '/work/forkert_lab/erik/MACAW/reshaped/slices-z'


for slices in range(99,-1,-1):
    new_number = slices + 41
    os.rename(path + '/reshaped_slice_' + str(slices) + '.npy',path + '/reshaped_slice_' + str(new_number) + '.npy')
