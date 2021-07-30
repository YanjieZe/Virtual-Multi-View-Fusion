import numpy as np
import torch.utils.data as data


class VirtualviewScannetDataset(data.Dataset):
    """
    TODO: Add virtual view
    """
    def __init__(self, cfg, mode='train', use_transform=True):
        self.cfg = cfg
        self.use_transform = use_transform
        self.mode=mode
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass

