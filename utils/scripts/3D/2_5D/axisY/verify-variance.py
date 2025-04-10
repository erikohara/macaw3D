import numpy as np
import pickle

pca_path = '/work/forkert_lab/erik/PCA/slices-y'

evalues = []

for slice in range(34,184):
    with open(pca_path + f"/evalues_slice_{slice}.pkl",'rb') as f:
        evalue = pickle.load(f)
    variance = evalue[-500:].sum()/evalue.sum()
    if variance < 0.995:
        print(f"Slice {slice} has variance {variance}")
