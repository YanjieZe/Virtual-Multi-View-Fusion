from PIL import Image
import numpy as np
import torch
import hydra

class Fusioner:
    """
    Get Mesh, then project them into 2D image
    """
    def __init__(self, point_cloud, intrinsic_depth, intrinsic_color):
        self.pc = point_cloud.unsqueeze(2)
        self.feature_vectors = np.zeros(self.pc.shape[0])
        self.intrinsic_depth = intrinsic_depth
        self.intrinsic_color = intrinsic_color
    

    def projection(self, depth_img, color_img, pose_matrix, semantic_label):
        """
        With depth check
        """
        # TODO: Is this extrinsic true?
        # # one method 
        # extrinsic = torch.inverse(pose_matrix)
        # another method
        extrinsic = pose_matrix
         
        rotation_matrix = extrinsic[:3,:3] # 3*3
        translation_vector = extrinsic[:3,3].unsqueeze(1) # 3*1
        translation_vector = (torch.ones(self.pc.shape[0])*translation_vector).T.unsqueeze(2)# num_point*3*1
        
        project_points = torch.matmul(rotation_matrix, self.pc) + translation_vector # num_point*3*1
        
        
        # TODO: 下面写的转化不太对？
        intrinsic_color = self.intrinsic_color[:3,:3]
        intrinsic_depth = self.intrinsic_depth[:3,:3]
        
        project_points_color = torch.matmul(intrinsic_color, project_points)
        project_points_depth = torch.matmul(intrinsic_depth, project_points)
        
        # depth that we use ex/intrinsic to get, use an average function to increase the precision
        depth_calcul = (project_points_depth[...,2,0] + project_points_color[...,2,0])/2
        
        # div Z
        project_points_depth[...,0,0] = project_points_depth[...,0,0]/torch.abs(project_points_depth[...,2,0])
        project_points_depth[...,1,0] = project_points_depth[...,1,0]/torch.abs(project_points_depth[...,2,0])
        project_points_depth[...,2,0] = project_points_depth[...,2,0]/torch.abs(project_points_depth[...,2,0])

        # div Z
        project_points_color[...,0,0] = project_points_color[...,0,0]/torch.abs(project_points_color[...,2,0])
        project_points_color[...,1,0] = project_points_color[...,1,0]/torch.abs(project_points_color[...,2,0])
        project_points_color[...,2,0] = project_points_color[...,2,0]/torch.abs(project_points_color[...,2,0])
        

        
        print(project_points_color[0])
        print(project_points_depth[0])
        
        # TODO: Align??
        print(color_img.shape)
        print(depth_img.shape)
        print('depth1', depth_calcul[0])
        print('depth2', depth_img[int(project_points_depth[0][0]/10)][int(project_points_depth[0][1]/10)])
        
        

@hydra.main(config_path='../config', config_name='config')
def demo(cfg):
    from realview_loader import RealviewScannetDataset
    dataset = RealviewScannetDataset(cfg, mode='train', use_transform=False)

    scene_idx = 0
    fetch_scene_data = dataset[scene_idx]
    pc = fetch_scene_data['mesh_vertices']
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
                        intrinsic_color=intrinsic_color)
    fusion_machine.projection(depth_img=depth_img, 
                            color_img=color_img,
                            pose_matrix=pose_matrix,
                            semantic_label=semantic_label
                            )

    # fusioner = Fusioner()



if __name__=='__main__':
    demo()
    
