"""
Usage:
from utils.fusion_2d_3d import Fusioner
"""
from PIL import Image
import numpy as np
import torch
import hydra
from tqdm import tqdm
import gc # garbage collector
import psutil 
import os


def print_memory_info():
    """
    Used for improving program
    """
    print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    info = psutil.virtual_memory()
    print( u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024) )
    print(u'当前使用的总内存占比：',info.percent)
    print(u'cpu个数：',psutil.cpu_count())


class Fusioner:
    """
    Get Mesh, then project them into 2D image, and get feature
    """
    def __init__(self, point_cloud, intrinsic_depth, intrinsic_color=None, collection_method='average', use_gt=False):
        self.pc = point_cloud.unsqueeze(2)

        self.feature_vector = None

        self.intrinsic_depth = intrinsic_depth
        self.intrinsic_color = intrinsic_color

        """
        Supported Methods for Feature Collection:
        'average': sum and average
        TODO: add other methods
        """
        self.collection_method = collection_method

        self.use_gt = use_gt
        self.counting_vector = torch.zeros(self.pc.shape[0]) # used for couting img


    def projection(self, depth_img,  pose_matrix, feature_img, color_img=None, threshold=1.0):
        """
        Implementation of 2D-3D Fusion Algorithm.
        Params:
            gt: If gt=True, feature img is semantic label img.
                If gt=False, feature img is the output of the model.

        FIXME: Get this function right
        """
        
        LargeNum = 5000
        device = depth_img.device
        
        # # one method 
        # extrinsic = torch.inverse(pose_matrix)

        # another method
        extrinsic = pose_matrix

        # ------------------------------------------------------------------------# 
        
        rotation_matrix = extrinsic[:3,:3] # 3*3
        translation_vector = extrinsic[:3,3].unsqueeze(1) # 3*1
        
        translation_vector = (torch.ones(self.pc.shape[0]).to(device)*translation_vector).T.unsqueeze(2)# num_point*3*1
        
        
        # FIXME: check the accuracy of code
        # intrinsic_color = self.intrinsic_color[:3,:4]
        intrinsic_depth = self.intrinsic_depth[:3,:4]
        
        extension = torch.ones(self.pc.shape[0]).unsqueeze(1).unsqueeze(1).to(device)

        # use extrinsic
        point_clouds = torch.hstack([self.pc, extension])
        project_points = torch.matmul(extrinsic, point_clouds)

        # use intrinsic
        # project_points_color = torch.matmul(intrinsic_color, project_points)
        project_points_depth = torch.matmul(intrinsic_depth, project_points)

        del project_points
        gc.collect()

        # get pixel level coordinates
        # project_points_color[...,0,0] = project_points_color[...,0,0]/torch.abs(project_points_color[...,2,0])
        # project_points_color[...,1,0] = project_points_color[...,1,0]/torch.abs(project_points_color[...,2,0])
        project_points_depth[...,0,0] = project_points_depth[...,0,0]/torch.abs(project_points_depth[...,2,0])
        project_points_depth[...,1,0] = project_points_depth[...,1,0]/torch.abs(project_points_depth[...,2,0])
        

        # compute depth in prediction
        depth_pred = torch.matmul(torch.inverse(rotation_matrix), translation_vector)
        depth_pred = torch.sqrt(torch.sum(torch.square(self.pc - depth_pred), dim=1)).squeeze(1)
        

        # clap the points which are not in the img
        
        project_points_depth = project_points_depth.squeeze(2)[...,0:2].long()
        row_bound = depth_img.shape[0]
        colum_bound = depth_img.shape[1]
        up_bound = torch.from_numpy(np.array([row_bound, colum_bound])).to(device)
        
        low_bound = torch.from_numpy(np.array([0, 0])).to(device)
        # bounded mask computation
        
        mask = project_points_depth<up_bound
        mask2 = project_points_depth>=low_bound
        
        mask = torch.sum(mask, dim=1)
        mask2 = torch.sum(mask2, dim=1)
        mask = mask + mask2
        mask = mask>=4 # get bounded mask
        
        del mask2 
        gc.collect()

        depth_real = torch.ones(self.pc.shape[0])*(LargeNum)
        depth_real = depth_real.to(device)
        
        
        # depth check
        
        # get depth
        if device!='cpu':
            mask = mask.cpu()
        # for i in tqdm(list(np.where(mask)[0]), desc='Get Real Depth'):
        for i in list(np.where(mask)[0]):
            depth_real[i] = depth_img[ project_points_depth[i][0], project_points_depth[i][1] ]
        

        # get depth mask
        depth_mask = torch.abs(depth_real - depth_pred) <= threshold

        # collect features

        if self.use_gt: # use ground truth label img
            
            if self.feature_vector is None:
                self.feature_vector = torch.zeros(self.pc.shape[0]).to(device)
            if self.collection_method == 'average':
                if device!='cpu':
                    depth_mask = depth_mask.cpu()
                # for i in tqdm(list(np.where(depth_mask)[0]),desc='Collect Features'):
                for i in list(np.where(depth_mask)[0]):
                    self.feature_vector[i] += feature_img[project_points_depth[i][0], project_points_depth[i][1] ]
                    
                    # count the img that have been used in projection
                    self.counting_vector[i] += 1
            else:#TODO: add other collection methods like K-nearset-neighbours
                
                raise Exception('Collection Method Error: Not Support %s'%self.collection_method)
       
        else: # use output of our model
            
            if self.collection_method == 'average':
                if self.feature_vector is None:
                    self.feature_vector = torch.zeros([self.pc.shape[0], feature_img.shape[0]]).to(device)
                feature_img = feature_img.permute(1,2,0)
                if device!='cpu':
                    depth_mask = depth_mask.cpu()
                # for i in tqdm(list(np.where(depth_mask)[0]),desc='Collect Features'):
                for i in list(np.where(depth_mask)[0]):
                    self.feature_vector[i] += feature_img[project_points_depth[i][0], project_points_depth[i][1] ]
                    self.counting_vector[i] += 1
            elif self.collection_method == 'knn':
                
                if self.feature_vector is None:
                    self.feature_vector = [ [] for i in range(self.pc.shape[0])]
                if device!='cpu':
                    depth_mask = depth_mask.cpu()
                for i in list(np.where(depth_mask)[0]):
                    self.feature_vector[i].append(feature_img[project_points_depth[i][0], project_points_depth[i][1]].max())
                    self.counting_vector[i] += 1
            else:
                raise Exception('Collection Method Error: Not Support %s'%self.collection_method)
                    
        
        

    def get_features(self):
        """
        Get the collected features
        """
        if self.collection_method == 'average':
            # for i in tqdm(list(np.where(self.counting_vector)[0]),desc='Get Avg Features'):
            for i in list(np.where(self.counting_vector)[0]):
                self.feature_vector[i] = self.feature_vector[i] / self.counting_vector[i]

            if self.use_gt:
                return self.feature_vector.long()
            else:
                return self.feature_vector
        elif self.collection_method == 'knn':

            self.feature_vector = [max(self.feature_vector[i], key=self.feature_vector[i].count)if self.feature_vector[i]!=[] else 0 for i in range(len(self.feature_vector))  ]
            return self.feature_vector
        else:
            raise Exception('Collection Method Error: Not Support %s'%self.collection_method)
       

        


@hydra.main(config_path='../config', config_name='config')
def demo(cfg):
    from realview_loader import RealviewScannetDataset
    dataset = RealviewScannetDataset(cfg, mode='train', use_transform=False)

    scene_idx = 0
    fetch_scene_data = dataset[scene_idx]
    pc = fetch_scene_data['point_cloud_xyz']
    vertices = fetch_scene_data['mesh_vertices']
   
    intrinsic_depth = fetch_scene_data['intrinsic_depth']
    intrinsic_color = fetch_scene_data['intrinsic_color']
    imgset = fetch_scene_data['imgset']

    img_idx = 0
    fetch_img_data = imgset[img_idx]
    depth_img = fetch_img_data['depth_img']
    color_img = fetch_img_data['color_img']
    pose_matrix = fetch_img_data['pose_matrix']
    semantic_label = fetch_img_data['semantic_label']
    
    fusion_machine = Fusioner(point_cloud=pc, 
                        intrinsic_depth=intrinsic_depth,
                        # intrinsic_color=intrinsic_color,
                        collection_method='average',
                        use_gt=True)
   
    fusion_machine.projection(depth_img=depth_img, 
                            # color_img=color_img,
                            pose_matrix=pose_matrix,
                            feature_img=semantic_label,
                            threshold=5.0)

    feature_vector = fusion_machine.get_features()

    print(feature_vector.max())


if __name__=='__main__':
    demo()
    
