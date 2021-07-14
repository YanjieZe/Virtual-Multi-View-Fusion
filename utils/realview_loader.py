import numpy as np
import torch
import torch.utils.data as data
import os


class ImageDataset(data.Dataset):
    
class RealviewScannetDataset(data.Dataset):
    def __init__(self, root_path='/data/ScanNetV2/scans', mode='train'):
        self.root_path = root_path
        self.mode = mode
        self.dir_list = os.listdir(root_path)
    
    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        scene_id = self.dir_list[index]
        image_path = os.path.join(self.root_path, scene_id, 'exported')
        print(image_path)


if __name__=='__main__':
    dataset = RealviewScannetDataset()
    dataset[0]

        
