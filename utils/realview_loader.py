import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
from torchvision.transforms import transforms
import sys


class ImageDataset(data.Dataset):
    def __init__(self, root_path, scene_id, image_form='color', transform=None):
        if image_form!='color' and image_form!='depth':
            raise Exception("Param Error: Only Support <color>/<depth> form")
        
        self.root_path = root_path
        self.scene_id = scene_id

        self.image_path = os.path.join(self.root_path, self.scene_id, 'exported',image_form)
        self.instance_path = os.path.join(self.root_path, self.scene_id, 'instance-filt')
        self.label_path = os.path.join(self.root_path, self.scene_id, 'label-filt')       

        # unzip file
        if not os.path.exists(self.label_path):
            zip_file_path = os.path.join(self.root_path, self.scene_id, '%s_2d-label-filt.zip'%(self.scene_id))
            os.system('unzip %s'%(zip_file_path))
        if not os.path.exists(self.instance_path):
            zip_file_path = os.path.join(self.root_path, self.scene_id, '%s_2d-instance-filt.zip'%(self.scene_id))
            os.system('unzip %s'%(zip_file_path))

        # store the file name of the images
        self.img_list = os.listdir(self.image_path)
        self.instance_list = os.listdir(self.instance_path)
        self.label_list = os.listdir(self.label_path)

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        image_name = os.path.join(self.image_path, self.img_list[index])
        instance_label_name = os.path.join(self.instance_path, self.instance_list[index])
        label_name = os.path.join(self.label_path, self.label_list[index])

        img = Image.open(image_name)
        img = self.transform(img)

        instance_label = Image.open(instance_label_name)
        instance_label = self.transform(instance_label)

        label_img = Image.open(label_name)
        label_img = self.transform(label_img)
        return img, instance_label, label_img

    
class RealviewScannetDataset(data.Dataset):
    def __init__(self, root_path='/data/ScanNetV2/scans', mode='train'):
        
        self.root_path = root_path
        self.mode = mode
        self.dir_list = os.listdir(root_path)
    
    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        scene_id = self.dir_list[index]

        color_imgset = ImageDataset(self.root_path, scene_id, 'color') # RGB image

        return color_imgset


if __name__=='__main__':
    dataset = RealviewScannetDataset()
    print(dataset[0][0][2])

        
