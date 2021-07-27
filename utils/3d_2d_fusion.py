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
    

    def projection(self, depth_img, color_img, pose_matrix, semantic_label, threshold=0.5):
        """
        With depth check
        TODO: 
        """
        # # one method 
        extrinsic = torch.inverse(pose_matrix)

        # another method
        # extrinsic = pose_matrix

        # ------------------------------------------------------------------------# 
        
        rotation_matrix = extrinsic[:3,:3] # 3*3
        translation_vector = extrinsic[:3,3].unsqueeze(1) # 3*1
        
        translation_vector = (torch.ones(self.pc.shape[0])*translation_vector).T.unsqueeze(2)# num_point*3*1
        
        
        
        # TODO: 下面写的转化不太对？
        intrinsic_color = self.intrinsic_color[:3,:4]
        intrinsic_depth = self.intrinsic_depth[:3,:4]
        
        extension = torch.ones(self.pc.shape[0]).unsqueeze(1).unsqueeze(1)
        
        # use extrinsic
        point_clouds = torch.hstack([self.pc, extension])
        project_points = torch.matmul(extrinsic, point_clouds)

        # use intrinsic
        project_points_color = torch.matmul(intrinsic_color, project_points)
        project_points_depth = torch.matmul(intrinsic_depth, project_points)

        # get pixel level coordinates
        project_points_color[...,0,0] = project_points_color[...,0,0]/torch.abs(project_points_color[...,2,0])
        project_points_color[...,1,0] = project_points_color[...,1,0]/torch.abs(project_points_color[...,2,0])
        project_points_depth[...,0,0] = project_points_depth[...,0,0]/torch.abs(project_points_depth[...,2,0])
        project_points_depth[...,1,0] = project_points_depth[...,1,0]/torch.abs(project_points_depth[...,2,0])
        
        print('depth coordinate', project_points_depth[0])
        print('color coordinate', project_points_color[0])

        # compute depth in prediction
        depth_pred = torch.matmul(torch.inverse(rotation_matrix), translation_vector)
        depth_pred = torch.sqrt(torch.sum(torch.square(self.pc - depth_pred), dim=1))

        
        print('depth img size', depth_img.shape, color_img.shape)
        print('depth in img', depth_img[int(project_points_depth[0][0]/10)][int(project_points_depth[0][1]/10)]) 
        print('depth pred', depth_pred[0])


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
                        intrinsic_color=intrinsic_color)
    fusion_machine.projection(depth_img=depth_img, 
                            color_img=color_img,
                            pose_matrix=pose_matrix,
                            semantic_label=semantic_label
                            )




if __name__=='__main__':
    demo()
    
