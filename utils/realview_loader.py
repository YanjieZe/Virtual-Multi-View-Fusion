import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
from torchvision.transforms import transforms
import sys
import hydra
try:
    from convert_scannet_instance_image import convert_instance_image
except:
    from utils.convert_scannet_instance_image import convert_instance_image

def collate_image(batch):
    """
    A rewrite for data loader's collate_fn
    Return: img, instance label, semantic label
    """
    if len(batch)==1:
        
        batch[0]['img'] = batch[0]['img'].unsqueeze(0)
        batch[0]['instance_label'] = batch[0]['instance_label'].unsqueeze(0)
        batch[0]['semantic_label'] = batch[0]['semantic_label'].unsqueeze(0)
        return batch[0]

    img = None
    semantic_label = None
    instance_label = None
    for i in range(len(batch)):
        if i==0:
            img = batch[0]['img'].unsqueeze(0)
            instance_label = batch[0]['instance_label'].unsqueeze(0)
            semantic_label = batch[0]['semantic_label'].unsqueeze(0)
        else:
            img = torch.vstack([img, batch[i]['img'].unsqueeze(0)])
            instance_label = torch.vstack([instance_label, batch[i]['instance_label'].unsqueeze(0)])
            semantic_label = torch.vstack([semantic_label, batch[i]['semantic_label'].unsqueeze(0)])
    grouping = dict()
    grouping['img'] = img
    grouping['instance_label'] = instance_label
    grouping['semantic_label'] = semantic_label

    return grouping

    
    

class ImageDataset(data.Dataset):
    """
    Dataset consist of Images in one Scene
    """
    def __init__(self, cfg, root_path, scene_id, transform=None):

        self.cfg = cfg

        image_form = cfg.dataset.image_form
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

        # read img
        img = Image.open(image_name)
        img = self.transform(np.array(img))

        # read full semantic label
        semantic_label = Image.open(label_name)
        semantic_label = self.transform(np.array(semantic_label))

        # convert instance image, since the original form can not be used directly
        instance_label = convert_instance_image(self.cfg.dataset.label_map, instance_label_name, label_name)
        instance_label = torch.from_numpy(instance_label.astype(np.int64))

        return {'img':img, 'instance_label':instance_label, 'semantic_label':semantic_label}
        
    
class RealviewScannetDataset(data.Dataset):
    """
    Dataset of Scenes
    """
    def __init__(self, cfg, mode='train'):

        self.cfg = cfg

        if mode=='train':
            self.root_path = cfg.dataset.train_path
        elif mode=='test':
            self.root_path = cfg.dataset.test_path
        else:
            raise Exception("Mode Error: Only Train/Test Supported")

        self.mode = mode
        self.dir_list = os.listdir(self.root_path)
        

    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        scene_id = self.dir_list[index]

        color_imgset = ImageDataset(self.cfg, self.root_path, scene_id) # RGB image

        return color_imgset


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    
    dataset = RealviewScannetDataset(cfg)
    print(dataset[0][0][1])

if __name__=='__main__':
    main()
    
    

        
