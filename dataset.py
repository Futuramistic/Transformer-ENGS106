import torch
from torch.utils.data import Dataset
import numpy as np

class TokenDataset(Dataset):
    def __init__(self, data, context_length, device):
        self.context_length = context_length
        self.device = device
        self.data = data

    def __len__(self):
        return len(self.data)-self.context_length

    def __getitem__(self, idx):
        x = torch.from_numpy((self.data[idx:idx+self.context_length]).astype(np.int64)).to(self.device)
        y = torch.from_numpy((self.data[idx+1:idx+1+self.context_length]).astype(np.int64)).to(self.device)
        return x, y