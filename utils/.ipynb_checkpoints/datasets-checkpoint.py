import numpy as np
import torch
from mnist import MNIST
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    # print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }


class MorphomnistDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        mndata = MNIST(root_dir)
        images, _ = mndata.load_training()
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = np.array(self.images[idx]).reshape(28, 28)
        # sample = sample[np.newaxis,:]
        if self.transform:
            sample = self.transform(sample)
        return sample
