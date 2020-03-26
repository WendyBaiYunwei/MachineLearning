import torch
from torch import utils
import os
import h5py
import numpy as np

class MyTestData(torch.utils.data.Dataset):
    def __init__(self, path, client):
        self.path = path
        self.client = client
        self.data = np.empty([1780, 3, 28, 28])
        self.label = np.empty([1780, 1])
        
        file = 'test' + str(self.client) + '.hdf5'
        path = os.path.join(self.path, file)
        
        with h5py.File(path, 'r') as h5_file:
            for i in range(1780):
                self.data[i] = h5_file['data'][i]
                self.label[i] = h5_file['label'][i]

    def __getitem__(self, i):
        data = self.data[i]  
        label = self.label[i]
        return (data, label)
                
    def __len__(self):
        return len(self.label)