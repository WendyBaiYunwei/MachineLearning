import torch
from torch import utils
import os
import h5py
import numpy as np

class MyData(torch.utils.data.Dataset):
    def __init__(self, path, client):
        self.path = path
        self.client = client
        self.data = np.empty([10840, 3, 28, 28])
        self.label = np.empty([10840, 1])
        
        file = 'train' + str(self.client) + '.hdf5'
        path = os.path.join(self.path, file)
        
        with h5py.File(path, 'r') as h5_file:
            for i in range(10840):
                self.data[i] = h5_file['data'][i]
                self.label[i] = h5_file['label'][i]
        
    def __getitem__(self, i):
        '''
        overflow = 0
        if (i+1)*self.batch_size > len(self.label):
            overflow = (i+1)*self.batch_size - len(self.label)
        '''
        #data = self.data[i*self.batch_size : (i+1)*self.batch_size - overflow]
        data = self.data[i]
        #label = self.label[i*self.batch_size : (i+1)*self.batch_size - overflow]
        label = self.label[i]
        return (data, label)
                
    def __len__(self):
        return len(self.label)