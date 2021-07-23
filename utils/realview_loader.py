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
    from export_trainmesh_for_evaluation import get_train_mesh
except:
    from utils.convert_scannet_instance_image import convert_instance_image
    from utils.export_trainmesh_for_evaluation import get_train_mesh

def collate_image(batch):
    """
    A rewrite for data loader's collate_fn
    Return: img, instance label, semantic label
    """
    if len(batch)==1:
        
        batch[0]['color_img'] = batch[0]['color_img'].unsqueeze(0)
        batch[0]['depth_img'] = label[0]['depth_img'].unsqueeze(0)
        batch[0]['instance_label'] = batch[0]['instance_label'].unsqueeze(0)
        batch[0]['semantic_label'] = batch[0]['semantic_label'].unsqueeze(0)
        batch[0]['pose_matrix'] = batch[0]['pose_matrix'].unsqueeze(0)
        return batch[0]

    img = None
    depth_img = None
    semantic_label = None
    instance_label = None
    pose_matrix = None
    for i in range(len(batch)):
        
        if i==0:

            img = batch[0]['color_img'].unsqueeze(0)
            depth_img = batch[0]['depth_img'].unsqueeze(0)
            instance_label = batch[0]['instance_label'].unsqueeze(0)
            semantic_label = batch[0]['semantic_label'].unsqueeze(0)
            pose_matrix = batch[0]['pose_matrix'].unsqueeze(0)

        else:

            img = torch.vstack([img, batch[i]['color_img'].unsqueeze(0)])
            depth_img = torch.vstack([depth_img, batch[i]['depth_img'].unsqueeze(0)])
            instance_label = torch.vstack([instance_label, batch[i]['instance_label'].unsqueeze(0)])
            semantic_label = torch.vstack([semantic_label, batch[i]['semantic_label'].unsqueeze(0)])
            pose_matrix = torch.vstack([pose_matrix, batch[i]['pose_matrix'].unsqueeze(0)])
    

    grouping = dict()
    grouping['color_img'] = img
    grouping['depth_img'] = depth_img
    grouping['instance_label'] = instance_label
    grouping['semantic_label'] = semantic_label
    grouping['pose_matrix'] = pose_matrix
 
    return grouping

   

class ImageDataset(data.Dataset):
    """
    Dataset consist of Images in one Scene
    """
    def __init__(self, cfg, root_path, scene_id, transform=None):

        self.cfg = cfg
        
        self.root_path = root_path
        self.scene_id = scene_id

        # several paths
        self.color_image_path = os.path.join(self.root_path, self.scene_id, 'exported','color')
        self.depth_image_path = os.path.join(self.root_path, self.scene_id, 'exported','depth')
        self.instance_path = os.path.join(self.root_path, self.scene_id, 'instance-filt')
        self.label_path = os.path.join(self.root_path, self.scene_id, 'label-filt')       
        self.pose_path = os.path.join(self.root_path, self.scene_id, 'exported', 'pose')       
        
        # unzip file
        
        if not os.path.exists(self.label_path):
            zip_file_path = os.path.join(self.root_path, self.scene_id, '%s_2d-label-filt.zip'%(self.scene_id))
            
            os.system('unzip %s'%(zip_file_path))
        if not os.path.exists(self.instance_path):
            zip_file_path = os.path.join(self.root_path, self.scene_id, '%s_2d-instance-filt.zip'%(self.scene_id))
            os.system('unzip %s'%(zip_file_path))

        # store the file name of the images
        self.color_img_list = os.listdir(self.color_image_path)
        self.depth_img_list = os.listdir(self.depth_image_path)
        self.instance_list = os.listdir(self.instance_path)
        self.label_list = os.listdir(self.label_path)
        self.pose_list = os.listdir(self.pose_path)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 缩放
            #transforms.RandomCrop(32, padding=4),  # 随机裁剪
            # transforms.ToTensor(),  
            #transforms.Normalize(0., 1.),  # 标准化均值为0标准差为1
            ])

        self.mapping = self.get_vaild_class_mapping()

    def __len__(self):
        return len(self.color_img_list)
    
    def __getitem__(self, index):
        # id check
        color_id = self.color_img_list[index].replace('.jpg', '')
        depth_id = self.depth_img_list[index].replace('.png', '')
        if color_id!=depth_id:
            raise Exception("ID Error: Color Image and Depth Image not match")
        
        # get path
        color_image_name = os.path.join(self.color_image_path, self.color_img_list[index])
        depth_image_name = os.path.join(self.depth_image_path, self.depth_img_list[index])
        instance_label_name = os.path.join(self.instance_path, self.instance_list[index])
        label_name = os.path.join(self.label_path, self.label_list[index])
        pose_file_name = os.path.join(self.pose_path, self.pose_list[index])
       
        # read color img
        img = Image.open(color_image_name)
        img = self.transform(img)
        img = torch.from_numpy(np.array(img).astype(np.float32).transpose(2,0,1))
        
        # read depth img
        depth_img = Image.open(depth_image_name)
        depth_img = self.transform(depth_img)
        depth_img = np.array(depth_img)
        depth_img = torch.from_numpy(depth_img)

        # read full semantic label
        semantic_label = Image.open(label_name)
        semantic_label = self.transform(semantic_label)
        semantic_label = torch.from_numpy(np.array(semantic_label))
        

        # convert instance image, since the original form can not be used directly
        instance_label = convert_instance_image(self.cfg.dataset.label_map, instance_label_name, label_name)
        instance_label = torch.from_numpy(instance_label.astype(np.int16))
        
        # get benchmark semantic label
        valid_class_id = self.cfg.valid_class_ids
        mask = torch.zeros_like(semantic_label).bool()
        for class_id in valid_class_id:
            mask = mask | (semantic_label==class_id)

        semantic_label_ = torch.zeros_like(semantic_label)
        semantic_label_[mask] = semantic_label[mask]
        semantic_label = semantic_label_

        def idx_map(idx):
            return self.mapping[idx]
        
        semantic_label = torch.from_numpy(np.array(list(map(idx_map, semantic_label)))).long()

        # get pose
        pose_matrix = self.get_pose_matrix(pose_file_name)


        return {'color_img': img, 
                'depth_img': depth_img,
                'pose_matrix': pose_matrix,
                'instance_label': instance_label, 
                'semantic_label': semantic_label}
        
    def get_vaild_class_mapping(self):
        valid_class_ids = self.cfg.valid_class_ids
        max_id = valid_class_ids[-1]
        mapping = np.ones(max_id+1)*valid_class_ids.index(max_id)
        
        for i in range(max_id+1):
            if i in valid_class_ids:
                mapping[i] = valid_class_ids.index(i)
            else:
                mapping[i] = valid_class_ids.index(max_id)

        return mapping


    def get_pose_matrix(self, pose_file_name):
        """
        read the pose matrix in file *.txt
        """
        with open(pose_file_name, 'r') as f:
            matrix = [[float(num) for num in line.split(' ')] for line in f]
        matrix = torch.from_numpy(np.array(matrix))
        return matrix
             
    def get_intrinsic_matrix(self, file_path):
        """
        read the intrinsic matrix in file *.txt
        """
        with open(file_path, 'r') as f:
            matrix = [[float(num) for num in line.split(' ')] for line in f]
        matrix = torch.from_numpy(np.array(matrix))
        return matrix 

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
        
        self.mapping = self.get_vaild_class_mapping()

    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        scene_id = self.dir_list[index]

        # get images, including color and depth
        imgset = ImageDataset(self.cfg, self.root_path, scene_id) 
        
        # get mesh
        mesh_file_path = os.path.join(self.root_path, scene_id, "%s_vh_clean_2.ply"%(scene_id))
        agg_file_path = os.path.join(self.root_path, scene_id,"%s.aggregation.json"%(scene_id)) 
        seg_file_path = os.path.join(self.root_path, scene_id, "%s_vh_clean_2.0.010000.segs.json"%(scene_id))
        label_map_file_path = self.cfg.dataset.label_map
        
        mesh_vertices, semantic_label, instance_label = get_train_mesh(mesh_file_path,
                            agg_file_path,
                            seg_file_path,
                            label_map_file_path,
                            type='instance')
        mesh_vertices = torch.from_numpy(mesh_vertices.astype(np.float64))
        # process the semantic label into the benchmark label
        semantic_label = torch.from_numpy(semantic_label.astype(np.int32))
        valid_class_id = self.cfg.valid_class_ids
        mask = torch.zeros_like(semantic_label).bool()
        for class_id in valid_class_id:
            mask = mask | (semantic_label==class_id)

        semantic_label_ = torch.zeros_like(semantic_label)
        semantic_label_[mask] = semantic_label[mask]
        semantic_label = semantic_label_

        def idx_map(idx):
            return self.mapping[idx]
        semantic_label = torch.from_numpy(np.array(list(map(idx_map, semantic_label)))).long()

        # get camera params
        intrinsic_path = os.path.join(self.root_path, scene_id, "exported", "intrinsic")
        intrinsic_color_matrix = self.get_intrinsic_matrix(os.path.join(intrinsic_path, "intrinsic_color.txt"))
        intrinsic_depth_matrix = self.get_intrinsic_matrix(os.path.join(intrinsic_path, "intrinsic_depth.txt"))
        

        return {'imgset':imgset, 
                'mesh_vertices':mesh_vertices, 
                'semantic_label':semantic_label, 
                'instance_label':instance_label,
                'intrinsic_color':intrinsic_color_matrix,
                'intrinsic_depth':intrinsic_depth_matrix}


    def get_vaild_class_mapping(self):
        valid_class_ids = self.cfg.valid_class_ids
        max_id = valid_class_ids[-1]
        mapping = np.ones(max_id+1)*valid_class_ids.index(max_id)
        
        for i in range(max_id+1):
            if i in valid_class_ids:
                mapping[i] = valid_class_ids.index(i)
            else:
                mapping[i] = valid_class_ids.index(max_id)
        
        return mapping

    def get_intrinsic_matrix(self, file_path):
        """
        read the intrinsic matrix in file *.txt
        """
        with open(file_path, 'r') as f:
            matrix = [[float(num) for num in line.split(' ')] for line in f]
        matrix = torch.from_numpy(np.array(matrix))
        return matrix

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    dataset = RealviewScannetDataset(cfg)
    

if __name__=='__main__':
    main()
    
    

        
